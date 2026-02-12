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

// Enable debug logging: 0 = off, 1 = on
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
    std::ofstream input_file;
    std::ofstream weight_file;
    std::ofstream psum_file;
#endif
#endif

public:
    SystolicArrayCore() {
        // Initialize accumulation buffer
        for (int pix = 0; pix < ACCUMULATION_BUFFER_SIZE; pix++)
            for (int j = 0; j < OC0; j++) accumulation_buffer[pix][j] = 0;

        // Initialize registers
        for (int i = 0; i < IC0; i++) {
            for (int j = 0; j < OC0+1; j++) input_reg[i][j] = 0;
            for (int j = 0; j < OC0; j++) input_reg2[i][j] = 0;
            for (int j = 0; j < OC0; j++) psum_reg2[i][j] = 0;
            for (int j = 0; j < OC0; j++) weight_reg[i][j] = 0;
        }
        for (int i = 0; i < IC0+1; i++)
            for (int j = 0; j < OC0; j++) psum_reg[i][j] = 0;

#if HLS_DEBUG
#ifndef __SYNTHESIS__
        input_file.open("input_file");
        weight_file.open("weight_file");
        psum_file.open("psum_file");
        if (!input_file.is_open() || !weight_file.is_open() || !psum_file.is_open()) {
            std::cerr << "Failed to open log files!" << std::endl;
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
    int debug_pixel_count = 0;
#endif

    while(paramsIn.available(1)) {
        Params params = paramsIn.read();
        uint_32 mac_iters = params.IC1 * params.FX * params.FY * params.OX0 * params.OY0;
        uint_32 ramp = IC0 + OC0 - 2;
        uint_32 total_steps = mac_iters + ramp;

        uint_16 ox0 = 0, oy0 = 0, fx = 0, fy = 0, ic1 = 0;

        for (uint_32 step = 0; step < IC1_MAX * FX_MAX * FY_MAX * OX0_MAX * OY0_MAX + IC0_MAX + OC0_MAX - 2; ++step) {
            if (step == total_steps) break;

            uint_32 pix = oy0 * params.OX0 + ox0;

            // -------------------------------
            // Load weights
            // -------------------------------
            if (step < mac_iters && ox0 == 0 && oy0 == 0) {
                for (int i = 0; i < IC0_MAX; i++) {
                    if (i < IC0) {
                        PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
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
            if (step < mac_iters) in_col = input.read();

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
                if (ic1 == 0 && fx == 0 && fy == 0) {
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        psum_buf.value[j].template set_val<AC_VAL_0>();
                        if (j == OC0 - 1) break;
                    }
                } else {
                    #pragma hls_unroll yes
                    for (int j = 0; j < OC0_MAX; j++) {
                        psum_buf.value[j] = accumulation_buffer[pix][j];
                        if (j == OC0 - 1) break;
                    }
                }
            }

            // -------------------------------
            // Debug logging
            // -------------------------------
#if HLS_DEBUG
#ifndef __SYNTHESIS__
            if (debug_pixel_count < 5 && step < mac_iters) {
                input_file << "Pixel " << debug_pixel_count << " Input: ";
                for (int i = 0; i < IC0; i++) input_file << int(in_col.value[i]) << " ";
                input_file << "\n";

                weight_file << "Pixel " << debug_pixel_count << " Weight: ";
                for (int i = 0; i < IC0; i++)
                    for (int j = 0; j < OC0; j++) weight_file << int(weight_reg[i][j]) << " ";
                weight_file << "\n";

                psum_file << "Pixel " << debug_pixel_count << " PSUM_in: ";
                for (int j = 0; j < OC0; j++) psum_file << int(psum_buf.value[j]) << " ";
                psum_file << "\n";
            }
#endif
#endif

            // -------------------------------
            // Accum FIFOs and PE array
            // -------------------------------
            PackedInt<OUTPUT_PRECISION, OC0> output_buf;

#define ACCUM_FIFO_BODY(z,i,unused) \
            ODTYPE BOOST_PP_CAT(psum_fifo_output_, i); \
            ODTYPE BOOST_PP_CAT(psum_fifo_input_, i) = psum_buf.value[i]; \
            BOOST_PP_CAT(psum_fifo_, i).run( BOOST_PP_CAT(psum_fifo_input_, i), BOOST_PP_CAT(psum_fifo_output_, i) ); \
            output_buf.value[i] = BOOST_PP_CAT(psum_fifo_output_, i);

            REPEAT(ACCUM_FIFO_BODY)

#pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) psum_reg[0][j] = output_buf.value[j];

#pragma hls_unroll yes
            for (int j = 0; j < OC0_MAX; j++) {
#pragma hls_unroll yes
                for (int i = 0; i < IC0_MAX; i++) {
                    pe[i][j].run(input_reg[i][j], psum_reg[i][j], weight_reg[i][j],
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
#pragma hls_unroll yes
                for (int i = 0; i < OC0_MAX; i++) {
                    accumulation_buffer[pix][i] = output_row.value[i];
                    if (i == OC0 - 1) break;
                }

                if (ic1 == params.IC1-1 && fx  == params.FX-1  && fy  == params.FY-1)
                    output.write(output_row);

#if HLS_DEBUG
#ifndef __SYNTHESIS__
                if (debug_pixel_count < 5) {
                    psum_file << "Pixel " << debug_pixel_count << " PSUM_out / Output: ";
                    for (int j = 0; j < OC0; j++) psum_file << int(output_row.value[j]) << " ";
                    psum_file << "\n";
                    debug_pixel_count++;
                }
#endif
#endif
            }

            // -------------------------------
            // Shift registers and counters
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

            if (++ox0 == params.OX0) { ox0 = 0; if (++oy0 == params.OY0) { oy0 = 0; if (++fy == params.FY) { fy = 0; if (++fx == params.FX) { fx = 0; ic1++; }}}}
        } // INNER_LOOP
    } // paramsIn loop
}

private:
    ProcessingElement<IDTYPE, WDTYPE, ODTYPE> pe[IC0][OC0];

    ODTYPE accumulation_buffer[ACCUMULATION_BUFFER_SIZE][OC0];
    WDTYPE weight_reg[IC0][OC0];
    IDTYPE input_reg[IC0][OC0+1];
    IDTYPE input_reg2[IC0][OC0];
    ODTYPE psum_reg[IC0+1][OC0];
    ODTYPE psum_reg2[IC0][OC0];

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
