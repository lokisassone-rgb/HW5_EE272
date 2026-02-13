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
         ac_channel<Params> &paramsOut)
    {

        #ifndef __SYNTHESIS__
        while (paramsIn.available(1))
        #endif
        {
        Params params = paramsIn.read();
        #pragma hls_pipeline_init_interval 1
        LABEL(xy_o) for (uint_16 p = 0; p < OX1_MAX * OY1_MAX; ++p) { //loop over image tiles
            LABEL(OC2) for(uint_16 oc1 = 0; oc1 < OC1_MAX; ++oc1){ // loop over kernel tiles
                paramsOut.write(params);
                if (oc1 == params.OC1 - 1) {
                    break;
                }
            }
            if (p == params.OX1 * params.OY1 - 1) {
                break;
            }
        }
        }
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
        systolicArrayLooper.run(paramsIn, paramsChannel);
        systolicArrayCore.run(input, weight, output, paramsChannel);
    }
private:
    SystolicArrayCore<IDTYPE, WDTYPE, ODTYPE, OC0, IC0> systolicArrayCore;
    SystolicArrayLooper systolicArrayLooper;
    ac_channel<Params> paramsChannel;
};

#endif
