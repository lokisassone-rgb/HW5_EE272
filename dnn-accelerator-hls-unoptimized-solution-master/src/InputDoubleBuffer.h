#ifndef INPUT_DOUBLE_BUFFER_H
#define INPUT_DOUBLE_BUFFER_H


template <int size, int IC0, int OC0>
class InputDoubleBufferWriter{
public:
    InputDoubleBufferWriter(){}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<Params> &paramsIn,
                        ac_channel<PackedInt<INPUT_PRECISION, 4> > &din,
                        ac_channel<chanStruct<PackedInt<INPUT_PRECISION,IC0>,size> > &dout)
    {
        #ifndef __SYNTHESIS__
        /* Note on ac_channel guards:
         * We expect to read numTiles * tileSize from din. Our C sim will run fine without
         * this guard since we explicitly write all the data before executing a kernel in the
         * testbench. However, it is good style and practice for writing modules in the future
         * since sometimes you cannot control when data will be written to a interface. For example,
         * if we decided connect our module to a memory simulation that writes din sporadically the
         * module will fail on a non-blocking din read in a C sim without the guard.
         */
        while (paramsIn.available(1) && din.available(
                    (paramsIn[0].OX1.to_int() * paramsIn[0].OY1.to_int() * 
                    ((paramsIn[0].OX0.to_int() - 1) * paramsIn[0].STRIDE.to_int() + paramsIn[0].FX.to_int()) *
                    ((paramsIn[0].OY0.to_int() - 1) * paramsIn[0].STRIDE.to_int() + paramsIn[0].FY).to_int() * paramsIn[0].IC1.to_int()) / 4))
        #endif
        {
            // -------------------------------
            // Your code starts here
            // -------------------------------

            Params params = paramsIn.read();
            ac_int<ac::log2_ceil<size+1>::val, false> tileSize = ((params.OX0 - 1) * params.STRIDE + params.FX) * 
                                ((params.OY0 - 1) * params.STRIDE + params.FY) * 
                                params.IC1;
            #pragma hls_pipeline_init_interval 1
            TILES: for (int t = 0; t < params.OX1 * params.OY1; t++) {
                chanStruct<PackedInt<INPUT_PRECISION,IC0>,size> tmp;

                // record one tile in buffer
                TILE: for (int i = 0; i < tileSize; i++) {
                    PackedInt<INPUT_PRECISION, IC0> memCol;  // one column in the memory
                    // each packet contains 4 values, pack IC0 tgt into one row
                    #pragma hls_unroll yes
                    for (int j = 0; j < IC0; j=j+4) {
                        PackedInt<INPUT_PRECISION, 4> packet = din.read();
                        #pragma hls_unroll yes
                        for (int k = 0; k < 4; k++) {
                            memCol.value[j+k] = packet.value[k];
                        }
                    }
                    tmp.data[i] = memCol;
                } // TILE
                // write a tile
                dout.write(tmp);
            } // TILES

            // -------------------------------
            // Your code ends here
            // -------------------------------
        }
    }
};

template <int size, int IC0, int OC0>
class InputDoubleBufferReader{
public:
    InputDoubleBufferReader(){}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<Params> &paramsIn,
                        ac_channel<chanStruct<PackedInt<INPUT_PRECISION, IC0>,size> > &din, 
                        ac_channel<PackedInt<INPUT_PRECISION, IC0> > &dout)
    {
        #ifndef __SYNTHESIS__
        while (paramsIn.available(1) && din.available(paramsIn[0].OX1.to_int() * paramsIn[0].OY1.to_int()))
        #endif
        {
            // -------------------------------
            // Your code starts here
            // -------------------------------

            Params params = paramsIn.read();
            uint_16 IX0 = (params.OX0 - 1) * params.STRIDE + params.FX;
            uint_16 IY0 = (params.OY0 - 1) * params.STRIDE + params.FY;

            #pragma hls_pipeline_init_interval 1
            TILES: for (int t = 0; t < params.OX1 * params.OY1; t++) {
                chanStruct<PackedInt<INPUT_PRECISION, IC0>,size> tmp;
                
                // read one tile from memory, and pass out one address at a time in the correct order
                tmp = din.read();
                // OC1 reuses
                OC1: for (int oc1 = 0; oc1 < params.OC1; oc1++) {
                    IC1: for (int ic1 = 0; ic1 < params.IC1; ic1++) {
                        FY: for (int fy = 0; fy < params.FY; fy++) {
                            FX: for (int fx = 0; fx < params.FX; fx++) {
                                OY0: for (int oy0 = 0; oy0 < params.OY0; oy0++) { 
                                    #pragma hls_pipeline_init_interval 1
                                    OX0: for (int ox0 = 0; ox0 < params.OX0; ox0++) { 
                                        uint_16 address = 
                                                params.STRIDE * ox0 + fx +
                                                (params.STRIDE * oy0 + fy) * IX0 +
                                                IY0 * IX0 * ic1;
                                        dout.write(tmp.data[address]);

                                    } // OX0
                                } // OY0
                            } // FX
                        } // FY
                    } // IC1
                } // OC1
            } // TILES

            // -------------------------------
            // Your code ends here
            // -------------------------------
        }
    }
};

template <int size, int IC0, int OC0>
class InputDoubleBuffer{
public:
  InputDoubleBuffer(){}

  #pragma hls_design interface
  void CCS_BLOCK(run)(ac_channel<PackedInt<INPUT_PRECISION, 4> > &inputs_in, 
                      ac_channel<PackedInt<INPUT_PRECISION, IC0> > &inputs_out,
                      ac_channel<Params> &paramsIn)
    {

        #ifndef __SYNTHESIS__
        while (paramsIn.available(1))
        #endif
        {
            Params params = paramsIn.read();

            inputDoubleBufferReaderParams.write(params);
            inputDoubleBufferWriterParams.write(params);

            inputDoubleBufferWriter.run(inputDoubleBufferWriterParams, inputs_in, mem);

            inputDoubleBufferReader.run(inputDoubleBufferReaderParams, mem, inputs_out);
        }
    }

private:
    ac_channel<chanStruct<PackedInt<INPUT_PRECISION, IC0>,size> > mem;
    
    InputDoubleBufferWriter<size, IC0, OC0> inputDoubleBufferWriter;
    ac_channel<Params> inputDoubleBufferWriterParams;
    
    InputDoubleBufferReader<size, IC0, OC0> inputDoubleBufferReader;
    ac_channel<Params> inputDoubleBufferReaderParams;
};

#endif
