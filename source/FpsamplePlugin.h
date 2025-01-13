#pragma once
#include <string>
#include <vector>

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

namespace nvinfer1 {

class FpsamplePlugin : public nvinfer1::IPluginV3,
                       public nvinfer1::IPluginV3OneCore,
                       public nvinfer1::IPluginV3OneBuild,
                       public nvinfer1::IPluginV3OneRuntime {

public:
    FpsamplePlugin( int32_t nsample );

    // 通过 IPluginV3 继承
    IPluginCapability* getCapabilityInterface( PluginCapabilityType type ) noexcept override;
    IPluginV3*         clone() noexcept override;

    // 通过 IPluginV3OneCore 继承
    AsciiChar const* getPluginName() const noexcept override;
    AsciiChar const* getPluginVersion() const noexcept override;
    AsciiChar const* getPluginNamespace() const noexcept override;

    // 通过 IPluginV3OneBuild 继承
    int32_t configurePlugin( DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
                             int32_t nbOutputs ) noexcept override;
    int32_t getOutputDataTypes( DataType* outputTypes, int32_t nbOutputs, const DataType* inputTypes,
                                int32_t nbInputs ) const noexcept override;
    int32_t getOutputShapes( DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
                             int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
                             IExprBuilder& exprBuilder ) noexcept override;
    size_t  getWorkspaceSize( DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                              DynamicPluginTensorDesc const* outputs, int32_t nbOutputs ) const noexcept override;
    bool    supportsFormatCombination( int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                       int32_t nbOutputs ) noexcept override;
    int32_t getNbOutputs() const noexcept override;

    // 通过 IPluginV3OneRuntime 继承
    int32_t onShapeChange( PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out,
                           int32_t nbOutputs ) noexcept override;
    int32_t enqueue( PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
                     void* const* outputs, void* workspace, cudaStream_t stream ) noexcept override;
    IPluginV3*                   attachToContext( IPluginResourceContext* context ) noexcept override;
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int32_t                  m_nsample;
    PluginFieldCollection    m_pfc;
    std::vector<PluginField> m_pluginAttributes;

    int32_t m_in_n  = 1;
    int32_t m_out_n = 1;
};

class FpsamplePluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FpsamplePluginCreator();

    // 通过 IPluginCreatorV3One 继承
    IPluginV3* createPlugin( AsciiChar const* name, PluginFieldCollection const* fc,
                             TensorRTPhase phase ) noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;
    AsciiChar const*             getPluginName() const noexcept override;
    AsciiChar const*             getPluginVersion() const noexcept override;
    AsciiChar const*             getPluginNamespace() const noexcept override;

private:
    PluginFieldCollection    m_pfc;
    std::vector<PluginField> m_pluginAttributes;
};
}  // namespace nvinfer1