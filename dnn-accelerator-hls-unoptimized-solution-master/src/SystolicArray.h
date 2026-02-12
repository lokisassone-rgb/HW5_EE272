#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include "ProcessingElement.h"
#include "conv.h"
#include "Fifo.h"
#include "SystolicArrayCore.h"

// Include mc_scverify.h for CCS_* macros
#include <mc_scverify.h>

class SystolicArrayLooper
{
public:
    SystolicArrayLooper() {}

#pragma hls_design interface
void run(ac_channel<Params> &paramsIn,
         ac_channel<Params> &paramsOut,
         ac_channel<LoopIndices> &loopIndicesOut)
    {
        // -------------------------------
        // Generate the loop indices here for the systolic array.
        // Write the loop indices as well as the params out to channels.
        // Your code starts here
        // -------------------------------
        #ifndef __SYNTHESIS__
        while (paramsIn.available(1))
        #endif
        {
        Params params = paramsIn.read();
        #pragma hls_pipeline_init_interval 1
        LABEL(xy_o) for (uint_16 p = 0; p < OX1_MAX * OY1_MAX; ++p) { //loop over image tiles        
            LABEL(OC2) for(uint_16 oc1 = 0; oc1 < OC1_MAX; ++oc1){ // loop over kernel tiles    
                LABEL(co) for (uint_16 ic1 = 0; ic1 < IC1_MAX; ++ic1) { // loop over channel tile
                    LABEL(winx) for (uint_16 fx = 0; fx < FX_MAX; ++fx) { // loop over filter window x
                        #pragma hls_pipeline_init_interval 1
                        LABEL(winy) for (uint_16 fy = 0; fy < FY_MAX; ++fy) { // loop over filter window y
                                LoopIndices loopIndices = {
                                    ic1, 
                                    fx, 
                                    fy
                                };
                                loopIndicesOut.write(loopIndices);
                                paramsOut.write(params);
                            if (fy == params.FY - 1) {
                                break;
                            }
                        }
                        if (fx == params.FX - 1) {
                            break;
                        }
                    }
                    if (ic1 == params.IC1 - 1) {
                        break;
                    }
                }
                if (oc1 == params.OC1 - 1) {
                    break;
                }
            }
            if (p == params.OX1 * params.OY1 - 1) {
                break;
            }
        }
        }
        // -------------------------------
        // Your code ends here
        // -------------------------------
    }
};

template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int OC0, int IC0>
class SystolicArrayWrapper
{
public:
    SystolicArrayWrapper(){}
    
#pragma hls_design interface
#pragma hls_pipeline_init_interval 1
    void run(ac_channel<PackedInt<INPUT_PRECISION, IC0> > &input, 
             ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &weight, 
             ac_channel<PackedInt<OUTPUT_PRECISION, OC0> > &output,
             ac_channel<Params> &paramsIn)
    {
        systolicArrayLooper.run(paramsIn, paramsChannel, loopIndicesChannel);
        systolicArrayCore.run(input, weight, output, paramsChannel, loopIndicesChannel);
    }
private:
    SystolicArrayCore<IDTYPE, WDTYPE, ODTYPE, OC0, IC0> systolicArrayCore;
    SystolicArrayLooper systolicArrayLooper;
    ac_channel<Params> paramsChannel;
    ac_channel<LoopIndices> loopIndicesChannel;
};

#endif
