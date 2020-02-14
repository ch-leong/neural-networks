#include <iostream>
#include <cassert>
#include "neuralnetworks.h"

template<typename T = NN_type>
inline static T RandomBetween(T min, T max)
{
    // Random float between -0.5 and 0.5.
    T rand1 = T(rand()) / T(RAND_MAX) - 0.5f;
    // Random float between min and max.
    return (max + min) / 2 + rand1 * (max - min);
}

NeuralNetworks::NeuralNetworks(unsigned const inputCount, std::vector<unsigned> const& hiddensCount
    , unsigned const outputCount, NN_type (*function)(std::vector<NN_type> const& scaledInput))
    : m_Function(function), m_InputCount(inputCount), m_HiddensCount(hiddensCount), m_OutputCount(outputCount)
    , m_TotalConnectionCount(0), m_InputLayer(), m_HiddenLayers(), m_OutputLayer()
{
    // Parameters check.
    assert(m_Function != nullptr);
    assert(m_InputCount != 0);
    for (unsigned const& hiddenCount : m_HiddensCount)
        assert(hiddenCount != 0);
    assert(m_OutputCount != 0);

    // Total connections between neurons count.
    unsigned prevCount = m_InputCount;
    for (unsigned const& hiddenCount : m_HiddensCount)
    {
        m_TotalConnectionCount += prevCount * hiddenCount;
        prevCount = hiddenCount;
    }
    m_TotalConnectionCount += prevCount * m_OutputCount;

    // Prepare for iteration.
    prevCount = m_InputCount;

    // Input layer memory allocation.
    m_InputLayer.resize(m_InputCount);

    // Hidden layer memory allocation and weights.
    m_HiddenLayers.resize(m_HiddensCount.size());
    for (unsigned i = 0; i < (unsigned)m_HiddenLayers.size(); ++i)
    {
        Layer& hiddenLayer = m_HiddenLayers[i];
        hiddenLayer.resize(m_HiddensCount[i]);
        for (Neuron& neuron : hiddenLayer)
        {
            neuron.resize(prevCount);
            for (NN_type& weight : neuron)
            {
                // Random NN_type between -1, 1.
                weight = RandomBetween(-1.f, 1.f);
            }
        }
        // For the next iteration.
        prevCount = m_HiddensCount[i];
    }

    // Output layer memory allocation and weights.
    m_OutputLayer.resize(m_OutputCount);
    for (Neuron& outputNeuron : m_OutputLayer)
    {
        outputNeuron.resize(prevCount);
        for (NN_type& weight : outputNeuron)
        {
            // Random NN_type between -1, 1.
            weight = RandomBetween(-1.f, 1.f);
        }
    }
}

std::vector<NN_type> NeuralNetworks::GetAllWeights() const
{
    Neuron weights;
    weights.reserve(m_TotalConnectionCount);

    // Hidden Layer.
    for (Layer const& hiddenLayer : m_HiddenLayers)
    {
        for (Neuron const& neuron : hiddenLayer)
        {
            for (NN_type const& weight : neuron)
            {
                weights.push_back(weight);
            }
        }
    }

    // Output Layer.
    for (Neuron const& neuron : m_OutputLayer)
    {
        for (NN_type const& weight : neuron)
        {
            weights.push_back(weight);
        }
    }

    return std::move(weights);
}

void NeuralNetworks::SetAllWeights(const std::vector<NN_type>& layerNeuronWeights)
{
    assert((unsigned)layerNeuronWeights.size() == m_TotalConnectionCount);
    unsigned index = 0;

    // Hidden Layer.
    for (Layer& hiddenLayer : m_HiddenLayers)
    {
        for (Neuron& neuron : hiddenLayer)
        {
            Neuron weights;
            for (NN_type& weight : neuron)
            {
                weights.push_back(layerNeuronWeights[index++]);
            }
            neuron = weights;
        }
    }

    // Output Layer.
    for (Neuron& neuron : m_OutputLayer)
    {
        Neuron weights;
        for (NN_type& weight : neuron)
        {
            weights.push_back(layerNeuronWeights[index++]);
        }
        neuron = weights;
    }
}

void NeuralNetworks::FeedForward(const std::vector<NN_type>& input, std::vector<NN_type>& output) const
{
    // Iterator.
    Neuron layerInput = input, layerOutput;

    // Hidden Layer.
    for (Layer const& hiddenLayer : m_HiddenLayers)
    {
        for (Neuron const& neuron : hiddenLayer)
        {
            layerOutput.push_back(GetWeightedSum(neuron, layerInput));
        }

        // Untill we reach the last hidden layer.
        layerInput = layerOutput;
        layerOutput.clear();
    }

    // Output Layer.
    for (Neuron const& neuron : m_OutputLayer)
    {
        layerOutput.push_back(GetWeightedSum(neuron, layerInput));
    }

    output = layerOutput;
}

NN_type NeuralNetworks::GetWeightedSum(std::vector<NN_type> const& weights, std::vector<NN_type> const& inputs) const
{
    Neuron weightedSum;
    for (unsigned i = 0; i < (unsigned)inputs.size(); ++i)
    {
        weightedSum.push_back(inputs[i] * weights[i]);
    }
    return m_Function(weightedSum);
}
