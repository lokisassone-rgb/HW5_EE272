#ifndef WEIGHT_DOUBLE_BUFFER_H
#define WEIGHT_DOUBLE_BUFFER_H


template <int size, int IC0, int OC0>
class WeightDoubleBufferWriter{
public:
    WeightDoubleBufferWriter(){}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<Params> &paramsIn,
                        ac_channel<PackedInt<WEIGHT_PRECISION, 4> > &din,
                        ac_channel<chanStruct<PackedInt<WEIGHT_PRECISION, OC0>, size> > &dout)
    {
        // -------------------------------
        // Your code starts here
        // -------------------------------

        #ifndef __SYNTHESIS__
        /* Note on ac_channel guards:
         * We expect to read numTiles * tileSize from din. Our C sim will run fine without
         * this guard since we explicitly write all the data before executing a kernel in the
         * testbench. However, it is good style and practice for writing modules in the future
         * since sometimes you cannot control when data will be written to a interface. For example,
         * if we decided connect our module to a memory simulation that writes din sporadically the
         * module will fail on a non-blocking din read in a C sim without the guard.
         */
        while (paramsIn.available(1) && din.available((paramsIn[0].OX1.to_int() * paramsIn[0].OY1.to_int() * paramsIn[0].OC1.to_int() *
                                                       paramsIn[0].IC1.to_int()*IC0*paramsIn[0].FX.to_int()*paramsIn[0].FY.to_int()) / 4))
        #endif
        {
            Params params = paramsIn.read();
            ac_int<ac::log2_ceil<size+1>::val, false> tileSize = FX_MAX * FY_MAX * IC0 * IC1_MAX;

            TILES: for (int t = 0; t < OX1_MAX * OY1_MAX * OC1_MAX; t++) {
                chanStruct<PackedInt<WEIGHT_PRECISION, OC0>,size> tmp;
                TILE: for (int i = 0; i < tileSize; i++) {
                    // each packet contains 4 values, pack OC0 tgt into one row
                    PackedInt<WEIGHT_PRECISION, OC0> memRow;  // one row in the memory
                    for (int j = 0; j < OC0; j=j+4) {
                        PackedInt<WEIGHT_PRECISION, 4> packet = din.read();

                        for (int k = 0; k < 4; k++) {
                            memRow.value[j+k] = packet.value[k];
                        }
                    }
                    tmp.data[i] = memRow;
                    if (i == params.FX * params.FY * IC0 * IC1_MAX - 1) {
                        break; // no more data to read for this tile
                    }
                }  // TILE
                dout.write(tmp);
                
                if(t == params.OX1 * params.OY1 * params.OC1 - 1) {
                    break; // no more tiles to read
                }
            } // TILES
        }

        // -------------------------------
        // Your code ends here
        // -------------------------------
    }
};

template <int size, int IC0, int OC0>
class WeightDoubleBufferReader{
public:
    WeightDoubleBufferReader(){}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<Params> &paramsIn,
                        ac_channel<chanStruct<PackedInt<WEIGHT_PRECISION, OC0>,size> > &din, 
                        ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &dout)
    {
        // -------------------------------
        // Your code starts here
        // -------------------------------

        #ifndef __SYNTHESIS__
        while (paramsIn.available(1))
        #endif
        {
            Params params = paramsIn.read();
            ac_int<ac::log2_ceil<size+1>::val, false> tileSize = FX_MAX * FY_MAX * IC0 * IC1_MAX;

            // read in new tile for every oc1
            TILES: for (int t = 0; t < OX1_MAX * OY1_MAX * OC1_MAX; t++) {
                chanStruct<PackedInt<WEIGHT_PRECISION, OC0>,size> tmp;
                tmp = din.read();
                TILE: for (int i = 0; i < tileSize; i++) {
                    dout.write(tmp.data[i]);
                    if (i == params.FX * params.FY * IC0 * params.IC1 - 1) {
                        break; // no more data to read for this tile
                    }
                } // TILE
                if (t == params.OX1 * params.OY1 * params.OC1 - 1) {
                    break; // no more tiles to read
                }
            } // TILES
        }

        // -------------------------------
        // Your code ends here
        // -------------------------------
    }
};

template <int size, int IC0, int OC0>
class WeightDoubleBuffer{
public:
  WeightDoubleBuffer(){}

  #pragma hls_design interface
  void CCS_BLOCK(run)(ac_channel<PackedInt<WEIGHT_PRECISION, 4> > &weights_in, 
                      ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &weights_out,
                      ac_channel<Params> &paramsIn)
    {
        #ifndef __SYNTHESIS__
        while (paramsIn.available(1))
        #endif
        {
            Params params = paramsIn.read();

            // #ifndef __SYNTHESIS__
            // ac_int<ac::log2_ceil<size>::val, false> block_size = IC0*params.OC1*params.FX*params.FY;
            // assert(block_size <= size);
            // #endif

            weightDoubleBufferReaderParams.write(params);
            weightDoubleBufferWriterParams.write(params);

            weightDoubleBufferWriter.run(weightDoubleBufferWriterParams, weights_in, mem);
            weightDoubleBufferReader.run(weightDoubleBufferReaderParams, mem, weights_out);
        }
    }

private:
    ac_channel<chanStruct<PackedInt<WEIGHT_PRECISION, OC0>,size> > mem;
    
    WeightDoubleBufferWriter<size, IC0, OC0> weightDoubleBufferWriter;
    ac_channel<Params> weightDoubleBufferWriterParams;
    
    WeightDoubleBufferReader<size, IC0, OC0> weightDoubleBufferReader;
    ac_channel<Params> weightDoubleBufferReaderParams;
};


#endif
