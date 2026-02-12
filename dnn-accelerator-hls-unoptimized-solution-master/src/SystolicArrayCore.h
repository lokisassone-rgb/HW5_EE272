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
    std::ofstream input_file;
    std::ofstream weight_file;
    std::ofstream psum_file;
#endif
#endif

public:
    SystolicArrayCore() {
        for (int pix = 0; pix < ACCUMULATION_BUFFER_SIZE; pix++) {
            for (int j = 0; j < OC0; j++) accumulation_buffer[pix][j] = 0;
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
        input_file.open("input_file");
        weight_file.open("weight_file");
        psum_file.open("psum_file");

        if (!input_file.is_open() || !weight_file.is_open() || !psum_file.is_open()) {
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
    // Read params once
    Params params = paramsIn.read();

    const uint_32 total_pixels = params.OX0 * params.OY0;
    const uint_32 mac_iters = params.IC1 * params.FX * params.FY * total_pixels;
    const uint_32 ramp = IC0 + OC0 - 2;
    const uint_32 total_steps = mac_iters + ramp;

    uint_16 ox0 = 0, oy0 = 0, fx = 0, fy = 0, ic1 = 0;

#ifndef __SYNTHESIS__
    int debug_pixel_count = 0;
#endif

    // Main loop: fully bounded for HLS
    for (uint_32 step = 0; step < total_steps; ++step)
    {
        uint_32 pix = oy0 * params.OX0 + ox0;

        // -------------------------------
        // Load weights
        // -------------------------------
        if (step < mac_iters && ox0 == 0 && oy0 == 0) {
            for (int i = 0; i < IC0; i++) {
#pragma hls_unroll yes
                for (int j = 0; j < OC0; j++) {
                    PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
                    weight_reg[i][j] = w_row.value[j];
                }
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
        for (int i = 0; i < IC0; i++) input_reg[i][0] = input_buf.value[i];

        // -------------------------------
        // Partial sum init / read
        // -------------------------------
        PackedInt<OUTPUT_PRECISION, OC0> psum_buf;
        if (step < mac_iters) {
            if (ic1 == 0 && fx == 0 && fy == 0) {
#pragma hls_unroll yes
                for (int j = 0; j < OC0; j++) psum_buf.value[j].template set_val<AC_VAL_0>();
            } else {
#pragma hls_unroll yes
                for (int j = 0; j < OC0; j++) psum_buf.value[j] = accumulation_buffer[pix][j];
            }
        }

#ifndef __SYNTHESIS__
        if (debug_pixel_count < 5 && step < mac_iters) {
            input_file << "Pixel " << debug_pixel_count << " Input: ";
            for (int i = 0; i < IC0; i++) input_file << int(in_col.value[i]) << " ";
            input_file << "\n";

            weight_file << "Pixel " << debug_pixel_count << " Weight: ";
            for (int i = 0; i < IC0; i++) {
                for (int j = 0; j < OC0; j++) weight_file << int(weight_reg[i][j]) << " ";
            }
            weight_file << "\n";

            psum_file << "Pixel " << debug_pixel_count << " PSUM_in: ";
            for (int j = 0; j < OC0; j++) psum_file << int(psum_buf.value[j]) << " ";
            psum_file << "\n";
        }
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
        for (int j = 0; j < OC0; j++) psum_reg[0][j] = output_buf.value[j];

#pragma hls_unroll yes
        for (int j = 0; j < OC0; j++) {
#pragma hls_unroll yes
            for (int i = 0; i < IC0; i++) {
                pe[i][j].run(input_reg[i][j], psum_reg[i][j], weight_reg[i][j],
                            input_reg2[i][j], psum_reg2[i][j]);
            }
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
            for (int i = 0; i < OC0; i++) accumulation_buffer[pix][i] = output_row.value[i];

            if (ic1 == params.IC1-1 && fx  == params.FX-1  && fy  == params.FY-1) {
                output.write(output_row);
            }

#ifndef __SYNTHESIS__
            if (debug_pixel_count < 5) {
                psum_file << "Pixel " << debug_pixel_count << " PSUM_out / Output: ";
                for (int j = 0; j < OC0; j++) psum_file << int(output_row.value[j]) << " ";
                psum_file << "\n";
                debug_pixel_count++;
            }
#endif
        }

        // -------------------------------
        // Shift registers and manual counter update
        // -------------------------------
#pragma hls_unroll yes
        for (int j = 0; j < OC0; j++) {
#pragma hls_unroll yes
            for (int i = 0; i < IC0; i++) {
                input_reg[i][j+1] = input_reg2[i][j];
                psum_reg[i+1][j]  = psum_reg2[i][j];
            }
        }

        // Update counters
        if (++ox0 == params.OX0) { ox0 = 0; if (++oy0 == params.OY0) { oy0 = 0; if (++fy == params.FY) { fy = 0; if (++fx == params.FX) { fx = 0; ic1++; }}}}
    }
}

// -------------------------------
// Internal buffers and PE array
// -------------------------------
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
