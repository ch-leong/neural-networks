#pragma once
#include <vector>

#ifndef NN_type
#define NN_type float
#endif

class NeuralNetworks
{
    template<typename T = NN_type>
    inline static float Summation(std::vector<T> const& inputs)
    {
        T sum = 0;
        for (T const& input : inputs)
            sum += input;
        return sum;
    }

    typedef std::vector<NN_type> Neuron;
    typedef std::vector<Neuron> Layer;
public:
                         NeuralNetworks(unsigned const inputCount, std::vector<unsigned> const& hiddensCount, unsigned const outputCount, NN_type (*function)(std::vector<NN_type> const& scaledInput) = Summation);
                         ~NeuralNetworks() = default;
    unsigned             GetNumInput() const { return m_InputCount; }
    unsigned             GetNumOutput() const { return m_OutputCount; }
    unsigned             GetAllWeightsCount() const { return m_TotalConnectionCount; }
    std::vector<NN_type> GetAllWeights() const;
    void                 SetAllWeights(std::vector<NN_type> const& layerNeuronWeights);
    void                 FeedForward(std::vector<NN_type> const& input, std::vector<NN_type>& output) const;
private:
    NN_type              GetWeightedSum(std::vector<NN_type> const& weights, std::vector<NN_type> const& inputs) const;
    NN_type (*const             m_Function)(std::vector<NN_type> const& scaledInput);
    // Size.
    unsigned const              m_InputCount;
    std::vector<unsigned> const m_HiddensCount;
    unsigned const              m_OutputCount;
    unsigned                    m_TotalConnectionCount;
    // Data.
    Layer                       m_InputLayer;
    std::vector<Layer>          m_HiddenLayers;
    Layer                       m_OutputLayer;
};
