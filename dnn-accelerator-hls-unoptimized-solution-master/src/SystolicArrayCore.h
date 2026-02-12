#ifndef SYSTOLIC_ARRAY_CORE_H
#define SYSTOLIC_ARRAY_CORE_H

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>

#include "ProcessingElement.h"
#include "Fifo.h"

// Define this macro for debug logging
#define HLS_DEBUG 0
#if HLS_DEBUG
#ifndef __SYNTHESIS__
#include <iostream>
#include <fstream>
#include <string>

// Only works for square arrays
template <typename T>
void log_matrix(std::ofstream& file, T* data, int iteration, int side_length) {
    file << "Iteration: " << iteration << '\n';
    for (int r = 0; r < side_length; r++) {
        for (int c = 0; c < side_length; c++) {
            file << int(data[r][c].to_int()) << ' ';
        }
        file << '\n';
    }
    file << '\n';
}
#endif
#endif


struct LoopIndices{
    uint_16 ic1_idx;
    uint_16 fx_idx;
    uint_16 fy_idx;
};



template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int OC0, int IC0>
class SystolicArrayCore
{
    #if HLS_DEBUG
    #ifndef __SYNTHESIS__
    // Create log file information
    std::ofstream input_file;
    std::ofstream weight_file;
    std::ofstream psum_file;
    #endif
    #endif


public:
    SystolicArrayCore() {
        // Initialize internal buffers to 0
        for (int pix = 0; pix < ACCUMULATION_BUFFER_SIZE; pix++) {
            for (int j = 0; j < OC0; j++) {
                accumulation_buffer[pix][j] = 0;
            }
        }

        for (int i = 0; i < IC0; i++) {
            for (int j = 0; j < OC0+1; j++) input_reg[i][j] = 0;
            for (int j = 0; j < OC0; j++) input_reg2[i][j] = 0;
            for (int j = 0; j < OC0; j++) psum_reg2[i][j] = 0;
            for (int j = 0; j < OC0; j++) weight_reg[i][j] = 0;
        }

        for (int i = 0; i < IC0+1; i++) {
            for (int j = 0; j < OC0; j++) psum_reg[i][j] = 0;
        }
        #if HLS_DEBUG
        #ifndef __SYNTHESIS__

        // Creates filenames
        std::string input_filename = "input_file";
        std::string weight_filename = "weight_file";
        std::string psum_filename = "psum_file";

        // Opens log files when debugging
        input_file.open(input_filename.c_str());
        weight_file.open(weight_filename.c_str());
        psum_file.open(psum_filename.c_str());
        bool open_success = true;
        open_success = open_success && input_file.is_open();
        open_success = open_success && weight_file.is_open();
        open_success = open_success && psum_file.is_open();

        if (!open_success) {
            std::cerr << "Failed to open one or more log files." << std::endl;
        }
        #endif
        #endif
    }

#pragma hls_design interface
    #pragma hls_pipeline_init_interval 1
    void CCS_BLOCK(run)(
        ac_channel<PackedInt<INPUT_PRECISION, IC0> > &input, 
        ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &weight, 
        ac_channel<PackedInt<OUTPUT_PRECISION, OC0> > &output,
        ac_channel<Params> &paramsIn)
    {
        #ifndef __SYNTHESIS__
        // assert(params.OX0 * params.OY0 <= ACCUMULATION_BUFFER_SIZE);
        #endif

        #ifndef __SYNTHESIS__
        while(paramsIn.available(1))
        #endif
        {
            // -------------------------------
            // Read in the params from the channel
            // -------------------------------
            Params params = paramsIn.read();

            #ifndef __SYNTHESIS__
            static int run_debug_idx = 0;
            int local_run_idx = run_debug_idx++;
            int debug_input_reads = 0;
            int debug_weight_reads = 0;
            int debug_output_writes = 0;
            #endif

            // -------------------------------
            // HW5 Section 5: Flatten all inner loops into one INNER_LOOP
            // Total MAC iterations = IC1 × FX × FY × OX0 × OY0
            // Total steps = mac_iters + ramp + flush
            // -------------------------------
           // -------------------------------
        // -------------------------------
        uint_32 tile_size = params.OX0 * params.OY0;
        uint_32 mac_iters = params.IC1 * params.FX * params.FY * tile_size;
        uint_32 ramp = IC0 + OC0 - 1;
        uint_32 total_steps = mac_iters + ramp;

        // Reset per-run architectural state to avoid cross-tile/oc1 contamination.
        #pragma hls_unroll yes
        for (int pix = 0; pix < ACCUMULATION_BUFFER_SIZE; pix++) {
            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                accumulation_buffer[pix][j].template set_val<AC_VAL_0>();
                if (j == OC0 - 1) break;
            }
            if (pix == tile_size - 1) break;
        }

        #pragma hls_unroll yes
        for (int i = 0; i < IC0_MAX; i++) {
            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX + 1; j++) {
                input_reg[i][j].template set_val<AC_VAL_0>();
                if (j == OC0) break;
            }
            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                input_reg2[i][j].template set_val<AC_VAL_0>();
                psum_reg2[i][j].template set_val<AC_VAL_0>();
                if (j == OC0 - 1) break;
            }
            if (i == IC0 - 1) break;
        }

        #pragma hls_unroll yes
        for (int i = 0; i < IC0_MAX + 1; i++) {
            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                psum_reg[i][j].template set_val<AC_VAL_0>();
                if (j == OC0 - 1) break;
            }
            if (i == IC0) break;
        }

        IDTYPE zero_input;
        zero_input.template set_val<AC_VAL_0>();
        ODTYPE zero_output;
        zero_output.template set_val<AC_VAL_0>();

        // Flush FIFO state so each run starts from a clean pipeline.
        for (int flush = 0; flush < IC0_MAX; ++flush) {
            #define FLUSH_INPUT_FIFO_BODY(z,i,unused) \
                { IDTYPE BOOST_PP_CAT(flush_input_out_, i); \
                  BOOST_PP_CAT(input_fifo_, i).run(zero_input, BOOST_PP_CAT(flush_input_out_, i)); }
            REPEAT(FLUSH_INPUT_FIFO_BODY)

            #define FLUSH_PSUM_FIFO_BODY(z,i,unused) \
                { ODTYPE BOOST_PP_CAT(flush_psum_out_, i); \
                  BOOST_PP_CAT(psum_fifo_, i).run(zero_output, BOOST_PP_CAT(flush_psum_out_, i)); }
            REPEAT(FLUSH_PSUM_FIFO_BODY)

            #define FLUSH_ACCUM_FIFO_BODY(z,i,unused) \
                { ODTYPE BOOST_PP_CAT(flush_accum_out_, i); \
                  BOOST_PP_CAT(accum_fifo_, i).run(zero_output, BOOST_PP_CAT(flush_accum_out_, i)); }
            REPEAT(FLUSH_ACCUM_FIFO_BODY)

            if (flush == IC0 - 1) break;
        }

        #pragma hls_pipeline_init_interval 1
        LABEL(INNER_LOOP)
        for (uint_32 step = 0;
            step < IC1_MAX * FX_MAX * FY_MAX * OX0_MAX * OY0_MAX
                + IC0_MAX + OC0_MAX - 1;
            ++step)
        {
            // Stop after required cycles
            if (step == total_steps) break;

            uint_32 input_mac = step;
            uint_32 input_pix = input_mac % tile_size;
            uint_32 input_group = input_mac / tile_size;

            uint_32 tmp_in = input_group;
            uint_16 in_fx = tmp_in % params.FX;
            tmp_in /= params.FX;
            uint_16 in_fy = tmp_in % params.FY;
            tmp_in /= params.FY;
            uint_16 in_ic1 = tmp_in;

            // -------------------------------
            // Load weights once per IC1×FX×FY
            // -------------------------------
            if (step < mac_iters && input_pix == 0) {
                for (int i = 0; i < IC0_MAX; i++) {
                    if (i < IC0) {
                        PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
                        #ifndef __SYNTHESIS__
                        debug_weight_reads++;
                        #endif
                        #pragma hls_unroll yes
                        for (int j = 0; j < OC0_MAX; j++) {
                            weight_reg[i][j] = w_row.value[j];
                            if (j == OC0 - 1) break;
                        }
                    }
                    if (i == IC0 - 1) break;
                }
            }

            // -------------------------------
            // Read input
            // -------------------------------
            PackedInt<INPUT_PRECISION, IC0> in_col;
            if (step < mac_iters) {
                in_col = input.read();
                #ifndef __SYNTHESIS__
                debug_input_reads++;
                #endif
            } else {
                #pragma hls_unroll yes
                for (int k = 0; k < IC0_MAX; k++) {
                    in_col.value[k].template set_val<AC_VAL_0>();
                    if (k == IC0 - 1) break;
                }
            }

            // -------------------------------
            // Input FIFOs
            // -------------------------------
            PackedInt<INPUT_PRECISION, IC0> input_buf;

            #define INPUT_FIFO_BODY(z,i,unused) \
                IDTYPE BOOST_PP_CAT(input_fifo_output_, i); \
                IDTYPE BOOST_PP_CAT(input_fifo_input_, i) = in_col.value[i]; \
                BOOST_PP_CAT(input_fifo_, i).run( BOOST_PP_CAT(input_fifo_input_, i), BOOST_PP_CAT(input_fifo_output_, i) ); \
                input_buf.value[i] = BOOST_PP_CAT(input_fifo_output_, i);

            REPEAT(INPUT_FIFO_BODY)

            #pragma hls_unroll yes
            for (int i = 0; i < IC0_MAX; i++) {
                input_reg[i][0] = input_buf.value[i];
                if (i == IC0 - 1) break;
            }

            // -------------------------------
            // Partial sum init / read
            // -------------------------------
            PackedInt<OUTPUT_PRECISION, OC0> psum_buf;

            if (step < mac_iters) {
                if (in_ic1 == 0 && in_fx == 0 && in_fy == 0) {
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        psum_buf.value[j].template set_val<AC_VAL_0>();
                        if (j == OC0 - 1) break;
                    }
                } else {
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        psum_buf.value[j] = accumulation_buffer[input_pix][j];
                        if (j == OC0 - 1) break;
                    }
                }
            } else {
                #pragma hls_unroll yes
                for (int j = 0; j < OC0_MAX; j++) {
                    psum_buf.value[j].template set_val<AC_VAL_0>();
                    if (j == OC0 - 1) break;
                }
            }

            // -------------------------------
            // Accum FIFOs
            // -------------------------------
            PackedInt<OUTPUT_PRECISION, OC0> output_buf;

            #define ACCUM_FIFO_BODY(z,i,unused) \
                ODTYPE BOOST_PP_CAT(psum_fifo_output_, i); \
                ODTYPE BOOST_PP_CAT(psum_fifo_input_, i) = psum_buf.value[i]; \
                BOOST_PP_CAT(psum_fifo_, i).run( BOOST_PP_CAT(psum_fifo_input_, i), BOOST_PP_CAT(psum_fifo_output_, i) ); \
                output_buf.value[i] = BOOST_PP_CAT(psum_fifo_output_, i);

            REPEAT(ACCUM_FIFO_BODY)

            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                psum_reg[0][j] = output_buf.value[j];
                if (j == OC0 - 1) break;
            }

            // -------------------------------
            // Run PE array
            // -------------------------------
            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                #pragma hls_unroll yes
                for (int i = 0; i < IC0_MAX; i++) {
                    pe[i][j].run(input_reg[i][j], psum_reg[i][j],
                                weight_reg[i][j],
                                input_reg2[i][j], psum_reg2[i][j]);
                    if (i == IC0 - 1) break;
                }
                if (j == OC0 - 1) break;
            }

            // -------------------------------
            // Output FIFOs
            // -------------------------------
            PackedInt<OUTPUT_PRECISION, OC0> output_row;

            #define FIFO_WRITE_BODY(z,i,unused) \
                ODTYPE BOOST_PP_CAT(accum_fifo_output_, i); \
                BOOST_PP_CAT(accum_fifo_, i).run( psum_reg[IC0][i], BOOST_PP_CAT(accum_fifo_output_, i) ); \
                output_row.value[i] = BOOST_PP_CAT(accum_fifo_output_, i);

            REPEAT(FIFO_WRITE_BODY)

            // -------------------------------
            // Write back / output
            // -------------------------------
            if (step >= ramp && step < mac_iters + ramp) {
                uint_32 output_mac = step - ramp;
                uint_32 output_pix = output_mac % tile_size;
                uint_32 output_group = output_mac / tile_size;

                uint_32 tmp_out = output_group;
                uint_16 out_fx = tmp_out % params.FX;
                tmp_out /= params.FX;
                uint_16 out_fy = tmp_out % params.FY;
                tmp_out /= params.FY;
                uint_16 out_ic1 = tmp_out;

                #pragma hls_unroll yes
                for (int i = 0; i < OC0_MAX; i++) {
                    accumulation_buffer[output_pix][i] = output_row.value[i];
                    if (i == OC0 - 1) break;
                }

                if (out_ic1 == params.IC1-1 &&
                    out_fy  == params.FY-1  &&
                    out_fx  == params.FX-1) {
                    output.write(output_row);
                    #ifndef __SYNTHESIS__
                    debug_output_writes++;
                    #endif
                }
            }

            // -------------------------------
            // Shift registers
            // -------------------------------
            #pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
                #pragma hls_unroll yes
                for (int i = 0; i < IC0_MAX; i++) {
                    input_reg[i][j+1] = input_reg2[i][j];
                    psum_reg[i+1][j]  = psum_reg2[i][j];
                    if (i == IC0 - 1) break;
                }
                if (j == OC0 - 1) break;
            }

            // Input/output indices are derived from step; no manual counter update needed.
        }

        #ifndef __SYNTHESIS__
        int expected_input_reads = (int)(params.IC1 * params.FX * params.FY * tile_size);
        int expected_weight_reads = (int)(params.IC1 * params.FX * params.FY * IC0);
        int expected_output_writes = (int)tile_size;
        if (debug_input_reads != expected_input_reads ||
            debug_weight_reads != expected_weight_reads ||
            debug_output_writes != expected_output_writes) {
            std::cout << "[CORE DEBUG] run=" << local_run_idx
                      << " input_reads=" << debug_input_reads << " (exp " << expected_input_reads << ")"
                      << " weight_reads=" << debug_weight_reads << " (exp " << expected_weight_reads << ")"
                      << " output_writes=" << debug_output_writes << " (exp " << expected_output_writes << ")"
                      << std::endl;
        }
        #endif

        }
    
        // Debug example:
        // printf("outputs written: %d\n", output.size());
    }

private:
    
    // -------------------------------
    // Create the following:
    //  - PE array
    //  - accumulation buffer
    //  - weight registers
    //  - input registers (two sets, one at the input of the PE and one at the output) 
    //  - psum registers (two sets, one at the input of the PE and one at the output) 
    // Your code starts here
    // -------------------------------
    ProcessingElement<IDTYPE, WDTYPE, ODTYPE> pe[IC0][OC0];

    ODTYPE accumulation_buffer[ACCUMULATION_BUFFER_SIZE][OC0];
    WDTYPE weight_reg[IC0][OC0];
    IDTYPE input_reg[IC0][OC0+1];
    IDTYPE input_reg2[IC0][OC0];
    ODTYPE psum_reg[IC0+1][OC0];
    ODTYPE psum_reg2[IC0][OC0];
    // -------------------------------
    // Your code ends here
    // -------------------------------
    

#define INPUT_FIFOS_INIT(z, i, unused) \
    Fifo<IDTYPE, i + 1> BOOST_PP_CAT(input_fifo_, i);

    REPEAT(INPUT_FIFOS_INIT)

#define ACCUM_FIFOS_INIT(z, i, unused) \
    Fifo<ODTYPE, i + 1> BOOST_PP_CAT(psum_fifo_, i);

    REPEAT(ACCUM_FIFOS_INIT)
    

#define OUTPUT_FIFOS_INIT(z, i, unused) \
    Fifo<ODTYPE, OC0 - i> BOOST_PP_CAT(accum_fifo_, i);
    
    REPEAT(OUTPUT_FIFOS_INIT)
};

#endif
