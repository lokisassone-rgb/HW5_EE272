#ifndef DESERIALIZER_H
#define DESERIALIZER_H

template<typename DTYPE_SERIAL, typename DTYPE, int n>
class Deserializer{
public:
    Deserializer(){}

#pragma hls_design interface
void CCS_BLOCK(run)(ac_channel<DTYPE_SERIAL> &inputChannel,
                    ac_channel<DTYPE> &outputChannel)
    {
        #ifndef __SYNTHESIS__
        while(inputChannel.available(1))
        #endif
        {
            DTYPE output;
            for(int i = 0; i < n; i++){
                output.value[i] = inputChannel.read();
            }
            outputChannel.write(output);

        }
    }
};

class ParamsDeserializer{
public:
    ParamsDeserializer(){}

#pragma hls_design interface
void CCS_BLOCK(run)(ac_channel<uint_16> &inputChannel,
                    ac_channel<Params> &outputChannel1,
                    ac_channel<Params> &outputChannel2,
                    ac_channel<Params> &outputChannel3,
                    ac_channel<Params> &outputChannel4
                    )
    {
        Params params;
        
        params.OY1 = inputChannel.read();
        params.OX1 = inputChannel.read();
        params.OY0 = inputChannel.read();
        params.OX0 = inputChannel.read();
        params.OC1 = inputChannel.read();
        params.IC1 = inputChannel.read();
        params.FX = inputChannel.read();
        params.FY = inputChannel.read();
        params.STRIDE = inputChannel.read();

        outputChannel1.write(params);
        outputChannel2.write(params);
        outputChannel3.write(params);
        for (int i = 0; i < params.OX1 * params.OY1 * params.OC1; i++) {
            outputChannel4.write(params);
        }
    }

};

#endif
