#include "FpsamplePlugin.h"

#include <cassert>
#include <iostream>
#include <memory>

#include "./cuda_impl.h"
// #include "../kdtree_fpsample/kdtree_fpsample.h"

namespace {
char const* const FPSAMPLE_PLUGIN_VERSION{ "1" };
char const* const FPSAMPLE_PLUGIN_NAME{ "KDTreeFpsample" };
char const* const FPSAMPLE_PLUGIN_NAMESPACE{ "" };
}  // namespace

nvinfer1::FpsamplePlugin::FpsamplePlugin( int32_t nsample )
    : m_nsample( nsample ) {
    m_pluginAttributes.clear();

    PluginField pf_nsample = { "nsample", &m_nsample, PluginFieldType::kINT32, 1 };
    m_pluginAttributes.emplace_back( pf_nsample );

    m_pfc.nbFields = m_pluginAttributes.size();
    m_pfc.fields   = m_pluginAttributes.data();
};

IPluginCapability* FpsamplePlugin::getCapabilityInterface( PluginCapabilityType type ) noexcept {
    try {
        if ( type == PluginCapabilityType::kBUILD ) {
            return static_cast<IPluginV3OneBuild*>( this );
        }
        if ( type == PluginCapabilityType::kRUNTIME ) {
            return static_cast<IPluginV3OneRuntime*>( this );
        }
        assert( type == PluginCapabilityType::kCORE );
        return static_cast<IPluginV3OneCore*>( this );
    }
    catch ( ... ) {
        // log error
    }
    return nullptr;
}

IPluginV3* FpsamplePlugin::clone() noexcept {
    return new FpsamplePlugin( m_nsample );
}

AsciiChar const* FpsamplePlugin::getPluginName() const noexcept {
    return FPSAMPLE_PLUGIN_NAME;
}

AsciiChar const* FpsamplePlugin::getPluginVersion() const noexcept {
    return FPSAMPLE_PLUGIN_VERSION;
}

AsciiChar const* FpsamplePlugin::getPluginNamespace() const noexcept {
    return FPSAMPLE_PLUGIN_NAMESPACE;
}

int32_t FpsamplePlugin::configurePlugin( DynamicPluginTensorDesc const* in, int32_t nbInputs,
                                         DynamicPluginTensorDesc const* out, int32_t nbOutputs ) noexcept {
    return 0;
}

int32_t FpsamplePlugin::getOutputDataTypes( DataType* outputTypes, int32_t nbOutputs, const DataType* inputTypes,
                                            int32_t nbInputs ) const noexcept {
    // 确保输入输出数量符合预期
    assert( nbInputs == m_in_n );
    assert( nbOutputs == m_out_n );

    // 检查输入类型是否为 float
    assert( inputTypes[ 0 ] == DataType::kFLOAT );

    // 设置输出类型为 int64
    outputTypes[ 0 ] = DataType::kINT64;

    return 0;
}

int32_t FpsamplePlugin::getOutputShapes( const DimsExprs* inputs, int32_t nbInputs, const DimsExprs* shapeInputs,
                                         int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
                                         IExprBuilder& exprBuilder ) noexcept {
    // 确保输入输出数量符合预期
    assert( nbInputs == m_in_n );
    assert( nbOutputs == m_out_n );

    // 输入张量的形状
    const DimsExprs& inputShape = inputs[ 0 ];

    // 设置输出张量的形状，假设为 [BATCH, NUM_POINTS]
    outputs[ 0 ].nbDims = 2;                                  // 输出维度数为 2
    outputs[ 0 ].d[ 0 ] = inputShape.d[ 0 ];                  // BATCH
    outputs[ 0 ].d[ 1 ] = exprBuilder.constant( m_nsample );  // NUM_POINTS

    return 0;  // 返回 0 表示成功
}

size_t nvinfer1::FpsamplePlugin::getWorkspaceSize( DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                                   DynamicPluginTensorDesc const* outputs,
                                                   int32_t                        nbOutputs ) const noexcept {
    const int32_t B = inputs[ 0 ].desc.dims.d[ 0 ];
    const int32_t N = inputs[ 0 ].desc.dims.d[ 1 ];

    return B * N * sizeof( float );
}

bool FpsamplePlugin::supportsFormatCombination( int32_t pos, const DynamicPluginTensorDesc* inOut, int32_t nbInputs,
                                                int32_t nbOutputs ) noexcept {
    // 确保 pos 在合法范围内
    assert( pos < ( nbInputs + nbOutputs ) );

    // 检查输入
    if ( pos == 0 ) {
        // 输入必须是 float 类型，且格式为线性
        return inOut[ pos ].desc.type == DataType::kFLOAT && inOut[ pos ].desc.format == TensorFormat::kLINEAR;
    }
    // 检查输出
    else if ( pos == 1 ) {
        // 输出必须是 int64 类型，且格式为线性
        return inOut[ pos ].desc.type == DataType::kINT64 && inOut[ pos ].desc.format == TensorFormat::kLINEAR;
    }

    return false;  // 其他情况不支持
}

int32_t FpsamplePlugin::getNbOutputs() const noexcept {
    return m_out_n;
}

int32_t FpsamplePlugin::onShapeChange( PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out,
                                       int32_t nbOutputs ) noexcept {
    return 0;
}

int32_t FpsamplePlugin::enqueue( PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                 void const* const* inputs, void* const* outputs, void* workspace,
                                 cudaStream_t stream ) noexcept {

    const float* xyz  = static_cast<const float*>( inputs[ 0 ] );
    int64_t*     idxs = static_cast<int64_t*>( outputs[ 0 ] );

    // 从PluginTensorDesc中提取维度信息
    int B = inputDesc[ 0 ].dims.d[ 0 ];
    int N = inputDesc[ 0 ].dims.d[ 1 ];
    int C = inputDesc[ 0 ].dims.d[ 2 ];

    int S = outputDesc[ 0 ].dims.d[ 1 ];

    // 计算所需的内存大小N
    size_t temp_size      = B * N;
    size_t temp_size_byte = temp_size * sizeof( float );

    if ( workspace == nullptr ) {
        return -1;
    }
    cudaError_t err = cudaMemset( workspace, 10000, temp_size_byte );
    if ( err != cudaSuccess ) {
        return -1;
    }

    // 调用内核函数
    furthest_point_sampling_kernel_wrapper( B, N, S, xyz, static_cast<float*>( workspace ), idxs, stream );

    // // 拷贝回主机方便调试
    // int64_t* idxs_d = new int64_t[ 100 ];
    // err             = cudaMemcpy( idxs_d, idxs, 100 * sizeof( int64_t ), cudaMemcpyDeviceToHost );
    // if ( err != cudaSuccess ) {
    //     return -1;
    // }
    // delete[] idxs_d;

    return 0;
}

IPluginV3* FpsamplePlugin::attachToContext( IPluginResourceContext* context ) noexcept {
    return clone();
}

PluginFieldCollection const* FpsamplePlugin::getFieldsToSerialize() noexcept {
    return &m_pfc;
}

/// /////////////////////////////////////////////////////
/// FpsamplePluginCreator

FpsamplePluginCreator::FpsamplePluginCreator() {
    m_pluginAttributes.clear();

    PluginField pf_nsample = { "nsample", nullptr, PluginFieldType::kINT32, 1 };
    m_pluginAttributes.emplace_back( pf_nsample );

    m_pfc.nbFields = m_pluginAttributes.size();
    m_pfc.fields   = m_pluginAttributes.data();
}

IPluginV3* FpsamplePluginCreator::createPlugin( AsciiChar const* name, PluginFieldCollection const* fc,
                                                TensorRTPhase phase ) noexcept {
    assert( fc->nbFields == 1 );
    assert( fc->fields[ 0 ].type == PluginFieldType::kINT32 );
    fc->fields[ 0 ].name;

    FpsamplePlugin* plugin = new FpsamplePlugin( *static_cast<int32_t const*>( fc->fields[ 0 ].data ) );

    return plugin;
}

PluginFieldCollection const* FpsamplePluginCreator::getFieldNames() noexcept {
    return &m_pfc;
}

AsciiChar const* FpsamplePluginCreator::getPluginName() const noexcept {
    return FPSAMPLE_PLUGIN_NAME;
}

AsciiChar const* FpsamplePluginCreator::getPluginVersion() const noexcept {
    return FPSAMPLE_PLUGIN_VERSION;
}

AsciiChar const* FpsamplePluginCreator::getPluginNamespace() const noexcept {
    return FPSAMPLE_PLUGIN_NAMESPACE;
}
