/* Copyright 2019 Argonne National Laboratory
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "am_mpi.h"

#define COMM_BASED_PAR 1 /* 0: tag-based parallelism */

#if COMM_BASED_PAR
#define NUM_COMMS 20
#endif

static MPI_Win g_am_win = MPI_WIN_NULL;
static void *g_am_base = NULL;
static __thread int thread_id = 0;
//static __thread int recv_thread_id = 0;
static __thread int am_seq = 0;
static Realm::atomic<unsigned int> num_threads(0);
//static Realm::atomic<unsigned int> recv_num_threads(0);
#if COMM_BASED_PAR
static unsigned char buf_recv_list[NUM_COMMS][AM_BUF_COUNT][1024];
static unsigned char *buf_recv[NUM_COMMS];
static MPI_Request req_recv_list[NUM_COMMS][AM_BUF_COUNT];
#else
static unsigned char buf_recv_list[AM_BUF_COUNT][1024];
static unsigned char *buf_recv = buf_recv_list[0];
static MPI_Request req_recv_list[AM_BUF_COUNT];
#endif
static int n_am_mult_recv = 5;
static int pre_initialized;
static int node_size;
static int node_this;
#if COMM_BASED_PAR
static MPI_Comm parapoint[NUM_COMMS];
static int i_recv_list[NUM_COMMS] = {0};
static int comm_start = 0;
#else
static MPI_Comm multi_vci_comm;
static MPI_Comm comm_medium;
int i_recv_list = 0;
#endif

namespace Realm {
namespace MPI {

#define AM_MSG_HEADER_SIZE 4 * sizeof(int)


void AM_Init(int *p_node_this, int *p_node_size)
{
    char *s;
    MPI_Info info;

    MPI_Initialized(&pre_initialized);
    if (pre_initialized) {
        int mpi_thread_model;
        MPI_Query_thread(&mpi_thread_model);
        assert(mpi_thread_model == MPI_THREAD_MULTIPLE);
    } else {
        int mpi_thread_model;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mpi_thread_model);
        assert(mpi_thread_model == MPI_THREAD_MULTIPLE);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &node_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_this);
    *p_node_size = node_size;
    *p_node_this = node_this;
    
    /* Request for VCIs */
    MPI_Info_create(&info);
    MPI_Info_set(info, "mpi_assert_new_vci", "true");
#if COMM_BASED_PAR
    MPI_Info_set(info, "mpi_num_vcis", "1");
    for (int i = 0; i < NUM_COMMS; i++) { /* TODO: do we know the number of threads here? */
        MPI_Comm_dup_with_info(MPI_COMM_WORLD, info, &parapoint[i]);
    }
#else
    MPI_Info_set(info, "mpi_assert_unordered", "true"); /* Can use tags for VCI selection */
    
    MPI_Info_set(info, "mpi_assert_tag_based_parallelism", "true"); // tag-based parallelism possible
    MPI_Info_set(info, "mpi_num_tag_bits_for_vci", "5"); // number of bits in the MPI tag to use for VCI selection
    MPI_Info_set(info, "mpi_num_tag_bits_for_app", "5"); // number of bits in the MPI tag for the app
    
    MPI_Info_set(info, "mpi_num_vcis", "10"); /* TODO: do we know the number of threads here? */
    /* TODO: hints to negotiate bits for VCI hashing and user tag*/
    MPI_Comm_dup_with_info(MPI_COMM_WORLD, info, &multi_vci_comm);
#endif
    
    s = getenv("AM_MULT_RECV");
    if (s) {
        n_am_mult_recv = atoi(s);
    }
    //printf("Rank %d: n_am_mult_recv is %d\n", node_this, n_am_mult_recv);
#if COMM_BASED_PAR
    for (int comm_i = 0; comm_i<NUM_COMMS; comm_i++) {
        for (int  i = 0; i<n_am_mult_recv; i++) {
            CHECK_MPI( MPI_Irecv(buf_recv_list[comm_i][i], 1024, MPI_CHAR, MPI_ANY_SOURCE, 0x1, parapoint[comm_i], &req_recv_list[comm_i][i]) );
        }
        buf_recv[comm_i] = buf_recv_list[comm_i][0];
    }
#else
    for (int  i = 0; i<n_am_mult_recv; i++) {
        CHECK_MPI( MPI_Irecv(buf_recv_list[i], 1024, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, multi_vci_comm, &req_recv_list[i]) );
    }
    MPI_Comm_dup(multi_vci_comm, &comm_medium); /* comm_medium inherits VCIs of multi_vci_comm */
#endif

    MPI_Info_free(&info);
    printf("Initialized!\n");
}

void AM_Finalize()
{
    MPI_Request req_final;

    CHECK_MPI( MPI_Ibarrier(MPI_COMM_WORLD, &req_final) );
    while (1) {
        int is_done;
        MPI_Status status;
        CHECK_MPI( MPI_Test(&req_final, &is_done, &status) );
        if (req_final == MPI_REQUEST_NULL) {
            break;
        }
        AMPoll();
    }

    AMPoll_cancel();
#if COMM_BASED_PAR
    for (int comm_i = 0; comm_i<NUM_COMMS; comm_i++) {
        CHECK_MPI( MPI_Comm_free(&parapoint[comm_i]) );
    }
#else
    CHECK_MPI( MPI_Comm_free(&comm_medium) );
    CHECK_MPI( MPI_Comm_free(&multi_vci_comm) );
#endif

    if (!pre_initialized) {
        MPI_Finalize();
    }
}

void AM_init_long_messages(MPI_Win win, void *am_base)
{
    g_am_win = win;
    g_am_base = am_base;
}

void AMPoll()
{
    struct AM_msg *msg;
    int tn_src;

    /*if (recv_thread_id == 0) {
        recv_thread_id = recv_num_threads.fetch_add_acqrel(1) + 1;
    }*/

#if COMM_BASED_PAR
    while (1) {
        int got_am;
        int comm_i;
        MPI_Comm this_parapoint;
        MPI_Status status;
        for (int i = 0; i<NUM_COMMS; i++) {
            got_am = 0;
            comm_i = (comm_start + i) % NUM_COMMS;
            //printf("Rank %d: Polling request %d on comm %d\n", node_this, i_recv_list[comm_i], comm_i);
            CHECK_MPI( MPI_Test(&req_recv_list[comm_i][i_recv_list[comm_i]], &got_am, &status) ); 
            if (got_am) {
                msg = (struct AM_msg *) buf_recv[comm_i];
                tn_src = status.MPI_SOURCE;
                this_parapoint = parapoint[comm_i];
                
                /* Update comm_start for next poll */
                comm_start = (comm_i + 1) % NUM_COMMS;

                break;
            }
        }
        if (!got_am) {
            break;
        }

        char *header;
        char *payload;
        int payload_type = 0;
        if (msg->type == 0) {
            header = msg->stuff;
            payload = msg->stuff + msg->header_size;
        } else if (msg->type == 1) {
            header = msg->stuff + 4;

            int msg_tag = *(int32_t *)(msg->stuff);
            payload = (char *) malloc(msg->payload_size);
            payload_type = 1;    // need_free;
            //printf("Rank %d, recv_thread %d: Receiving msg type 1\n", node_this, recv_thread_id);
            //printf("Rank %d: Receiving payload from %d on comm %d, tag %d\n", node_this, tn_src, comm_i, msg_tag);
            CHECK_MPI( MPI_Recv(payload, msg->payload_size, MPI_BYTE, tn_src, msg_tag, this_parapoint, &status) );
            //printf("Rank %d: DONE Receiving payload from %d on comm %d, tag %d\n", node_this, tn_src, comm_i, msg_tag);
        } else if (msg->type == 2) {
            int offset = *(int32_t *)(msg->stuff);
            header = msg->stuff + 4;
            payload = (char *) g_am_base + offset;
        } else {
            assert(0 && "invalid message type");
            header = 0;
            payload = 0;
            msg->msgid = 0x7fff; // skips handler below
        }

        if (msg->msgid != 0x7fff) {
            Realm::ActiveMessageHandlerTable::MessageHandler handler = Realm::activemsg_handler_table.lookup_message_handler(msg->msgid);
            (*handler) (tn_src, header, payload, msg->payload_size);
        }

        if (payload_type == 1) {
            free(payload);
        }
        //printf("Rank %d, recv_thread %d: Posting replacement ANY_TAG receive\n", node_this, recv_thread_id);
        CHECK_MPI( MPI_Irecv(buf_recv_list[comm_i][i_recv_list[comm_i]], 1024, MPI_CHAR, MPI_ANY_SOURCE, 0x1, parapoint[comm_i], &req_recv_list[comm_i][i_recv_list[comm_i]]) );
        i_recv_list[comm_i] = (i_recv_list[comm_i] + 1) % n_am_mult_recv;
        buf_recv[comm_i] = buf_recv_list[comm_i][i_recv_list[comm_i]];
    }
#else
    while (1) {
        int got_am;
        MPI_Status status;
        CHECK_MPI( MPI_Test(&req_recv_list[i_recv_list], &got_am, &status) );
        if (!got_am) {
            break;
        }
        msg = (struct AM_msg *) buf_recv;

        tn_src = status.MPI_SOURCE;

        char *header;
        char *payload;
        int payload_type = 0;
        if (msg->type == 0) {
            header = msg->stuff;
            payload = msg->stuff + msg->header_size;
        } else if (msg->type == 1) {
            header = msg->stuff + 4;

            int msg_tag = *(int32_t *)(msg->stuff);
            payload = (char *) malloc(msg->payload_size);
            payload_type = 1;    // need_free;
            //printf("Rank %d, recv_thread %d: Receiving msg type 1\n", node_this, recv_thread_id);
            CHECK_MPI( MPI_Recv(payload, msg->payload_size, MPI_BYTE, tn_src, msg_tag, comm_medium, &status) );
        } else if (msg->type == 2) {
            int offset = *(int32_t *)(msg->stuff);
            header = msg->stuff + 4;
            payload = (char *) g_am_base + offset;
        } else {
            assert(0 && "invalid message type");
            header = 0;
            payload = 0;
            msg->msgid = 0x7fff; // skips handler below
        }

        if (msg->msgid != 0x7fff) {
            Realm::ActiveMessageHandlerTable::MessageHandler handler = Realm::activemsg_handler_table.lookup_message_handler(msg->msgid);
            (*handler) (tn_src, header, payload, msg->payload_size);
        }

        if (payload_type == 1) {
            free(payload);
        }
        //printf("Rank %d, recv_thread %d: Posting replacement ANY_TAG receive\n", node_this, recv_thread_id);
        CHECK_MPI( MPI_Irecv(buf_recv_list[i_recv_list], 1024, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, multi_vci_comm, &req_recv_list[i_recv_list]) );
        i_recv_list = (i_recv_list + 1) % n_am_mult_recv;
        buf_recv = buf_recv_list[i_recv_list];
    }
#endif

}

void AMPoll_cancel()
{
#if COMM_BASED_PAR
    for (int comm_i = 0; comm_i<NUM_COMMS; comm_i++) {
        for (int  i = 0; i<n_am_mult_recv; i++) {
            CHECK_MPI( MPI_Cancel(&req_recv_list[comm_i][i]) );
        }
    }
#else
    for (int  i = 0; i<n_am_mult_recv; i++) {
        CHECK_MPI( MPI_Cancel(&req_recv_list[i]) );
    }
#endif
}

void AMSend(int tgt, int msgid, int header_size, int payload_size, const char *header, const char *payload, int has_dest, MPI_Aint dest)
{
    char buf_send[1024];

    struct AM_msg *msg = (struct AM_msg *)(buf_send);
    msg->msgid = msgid;
    msg->header_size = header_size;
    msg->payload_size = payload_size;
    char *msg_header = msg->stuff;

    if (thread_id == 0) {
        thread_id = num_threads.fetch_add_acqrel(1) + 1;
    }

    if (has_dest) {
        assert(g_am_win);
        //printf("Rank %d, thread %d: AMSend with has_dest doing a Put \n", node_this, thread_id);
        CHECK_MPI( MPI_Put(payload, payload_size, MPI_BYTE, tgt, dest, payload_size, MPI_BYTE, g_am_win) );
        CHECK_MPI( MPI_Win_flush(tgt, g_am_win) );

        msg->type = 2;
        *((int32_t *) msg_header) = (int32_t) dest;
        memcpy(msg_header + 4, header, header_size);
        int n = AM_MSG_HEADER_SIZE + 4 + header_size;
        assert(tgt != node_this);
        //printf("Rank %d, thread %d: AMSend sending a message of type 2 after a Put-Win_flush\n", node_this, thread_id);

        //printf("Rank %d: Sending to %d with dest using thread %d on comm %d\n", node_this, tgt, thread_id, (thread_id-1)%NUM_COMMS);
#if COMM_BASED_PAR
        CHECK_MPI( MPI_Send(buf_send, n, MPI_BYTE, tgt, 0x1, parapoint[(thread_id-1)%NUM_COMMS]) );
#else
        int msg_tag = (thread_id << 10) + 1; /* TODO: use variable for number of bits to shift */
        CHECK_MPI( MPI_Send(buf_send, n, MPI_BYTE, tgt, msg_tag, multi_vci_comm) );
#endif
        //printf("Rank %d: DONE Sending to %d with dest using thread %d on comm %d\n", node_this, tgt, thread_id, (thread_id-1)%NUM_COMMS);
    } else if (AM_MSG_HEADER_SIZE + header_size + payload_size < 1024) {
        msg->type = 0;
        if (header_size > 0) {
            memcpy(msg_header, header, header_size);
        }
        if (payload_size > 0) {
            memcpy(msg_header + header_size, payload, payload_size);
        }
        int n = AM_MSG_HEADER_SIZE + header_size + payload_size;
        assert(tgt != node_this);
        //printf("Rank %d, thread %d: AMSend sending a message of type 0\n", node_this, thread_id);
        //printf("Rank %d: Sending to %d msg type 0 using thread %d on comm %d\n", node_this, tgt, thread_id, (thread_id-1)%NUM_COMMS);
#if COMM_BASED_PAR
        CHECK_MPI( MPI_Send(buf_send, n, MPI_CHAR, tgt, 0x1, parapoint[(thread_id-1)%NUM_COMMS]) );
#else
        int msg_tag = (thread_id << 10) + 1; /* TODO: use variable for number of bits to shift */
        CHECK_MPI( MPI_Send(buf_send, n, MPI_CHAR, tgt, msg_tag, multi_vci_comm) );
#endif
        //printf("Rank %d: DONE Sending to %d msg type 0 using thread %d on comm %d\n", node_this, tgt, thread_id, (thread_id-1)%NUM_COMMS);
    } else {
        msg->type = 1;
        int msg_tag = 0x0;
        //if (thread_id == 0) {
        //    thread_id = num_threads.fetch_add_acqrel(1) + 1;
        //}
        am_seq = (am_seq + 1) & 0x1f;
        msg_tag = (thread_id << 10) + am_seq;
        
        *((int32_t *) msg_header) = (int32_t) msg_tag;
        memcpy(msg_header + 4, header, header_size);
        int n = AM_MSG_HEADER_SIZE + 4 + header_size;
        assert(tgt != node_this);
        //printf("Rank %d, thread %d: AMSend sending a message of type 1; first send\n", node_this, thread_id);
        //printf("Rank %d: Sending to %d msg type 1 header using thread %d on comm %d\n", node_this, tgt, thread_id, (thread_id-1)%NUM_COMMS);
#if COMM_BASED_PAR
        CHECK_MPI( MPI_Send(buf_send, n, MPI_BYTE, tgt, 0x1, parapoint[(thread_id-1)%NUM_COMMS]) );
#else
        CHECK_MPI( MPI_Send(buf_send, n, MPI_BYTE, tgt, msg_tag, multi_vci_comm) );
#endif
        //printf("Rank %d: DONE Sending to %d msg type 1 header using thread %d on comm %d\n", node_this, tgt, thread_id, (thread_id-1)%NUM_COMMS);
        assert(tgt != node_this);
        //printf("Rank %d, thread %d: AMSend sending a message of type 1; second send on comm_medium\n", node_this, thread_id);
        //printf("Rank %d: Sending to %d msg type 1 payload using thread %d on comm %d, tag %d\n", node_this, tgt, thread_id, (thread_id-1)%NUM_COMMS, msg_tag);
#if COMM_BASED_PAR
        CHECK_MPI( MPI_Send(payload, payload_size, MPI_BYTE, tgt, msg_tag, parapoint[(thread_id-1)%NUM_COMMS]) );
#else
        CHECK_MPI( MPI_Send(payload, payload_size, MPI_BYTE, tgt, msg_tag, comm_medium) );
#endif
        //printf("Rank %d: DONE Sending to %d msg type 1 payload using thread %d on comm %d, tag %d\n", node_this, tgt, thread_id, (thread_id-1)%NUM_COMMS, msg_tag);
    }
}

} /* namespace MPI */
} /* namespace Realm */
