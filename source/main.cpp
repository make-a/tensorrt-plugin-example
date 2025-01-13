
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <memory>
#include <random>

#include "FpsamplePlugin.h"

using namespace nvinfer1;

class Logger : public ILogger {
    void log( Severity severity, const char* msg ) noexcept override {
        if ( severity != Severity::kINFO ) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

// 在Logger类之后，main函数之前添加
const char* layerTypeToString( nvinfer1::LayerType type ) {
    switch ( type ) {
    case LayerType::kCONVOLUTION:
        return "CONVOLUTION";
    case LayerType::kACTIVATION:
        return "ACTIVATION";
    case LayerType::kPOOLING:
        return "POOLING";
    case LayerType::kPLUGIN:
        return "PLUGIN";
    case LayerType::kPLUGIN_V2:
        return "PLUGIN_V2";
    case LayerType::kPLUGIN_V3:
        return "PLUGIN_V3";
    default:
        return "UNKNOWN";
    }
}

// 生成随机点云数据
void generateRandomPointCloud( float* data, int batch_size, int num_points, int channels ) {
    std::random_device                    rd;
    std::mt19937                          gen( rd() );
    std::uniform_real_distribution<float> dis( -1.0, 1.0 );

    for ( int b = 0; b < batch_size; b++ ) {
        for ( int n = 0; n < num_points; n++ ) {
            for ( int c = 0; c < channels; c++ ) {
                data[ b * num_points * channels + n * channels + c ] = dis( gen );
            }
        }
    }
}

void CreateFpsampleLayer ( INetworkDefinition * network, int batchSize, int numPoints, int channels, int numSamples ) {

}

int main() {
    // 初始化CUDA运行时
    cudaError_t error = cudaSetDevice( 0 );
    if ( error != cudaSuccess ) {
        std::cerr << "Failed to initialize CUDA: " << cudaGetErrorString( error ) << std::endl;
        return -1;
    }

    // 创建builder和network
    std::unique_ptr<IBuilder> builder( createInferBuilder( gLogger ) );
    if ( !builder ) {
        std::cerr << "Failed to create builder" << std::endl;
        return -1;
    }

    uint32_t                            flag = 1U << static_cast<uint32_t>( NetworkDefinitionCreationFlag::kEXPLICIT_BATCH );
    std::unique_ptr<INetworkDefinition> network( builder->createNetworkV2( flag ) );
    if ( !network ) {
        std::cerr << "Failed to create network" << std::endl;
        return -1;
    }

    // 注册插件
    auto creator = new FpsamplePluginCreator();
    getPluginRegistry()->registerCreator( *creator, "" );

    // 设置输入维度
    const int batchSize  = 1;
    const int numPoints  = 1024;
    const int channels   = 3;
    const int numSamples = 512;

    // 创建输入tensor
    ITensor* input = network->addInput( "input", DataType::kFLOAT, Dims3( batchSize, numPoints, channels ) );

    // 创建插件层
    PluginField nsampleField( "nsample", &numSamples, PluginFieldType::kINT32, 1 );

    PluginFieldCollection fc;
    fc.nbFields = 1;
    fc.fields   = &nsampleField;

    auto     plugin         = creator->createPlugin( "fpsample", &fc, TensorRTPhase::kBUILD );
    ITensor* inputTensors[] = { input };
    auto     layer          = network->addPluginV3( inputTensors, 1, nullptr, 0, *plugin );

    // 标记输出
    network->markOutput( *layer->getOutput( 0 ) );

    // 打印网络结构
    std::cout << "\n\n\n=== Network Structure ===" << std::endl;
    for ( int i = 0; i < network->getNbLayers(); i++ ) {
        auto layer = network->getLayer( i );
        std::cout << "Layer " << i << ": " << layer->getName() << " (Type: " << layerTypeToString( layer->getType() ) << ")"
                  << std::endl;

        // 打印输入维度
        std::cout << "  Inputs: ";
        for ( int j = 0; j < layer->getNbInputs(); j++ ) {
            auto dims = layer->getInput( j )->getDimensions();
            std::cout << "[";
            for ( int k = 0; k < dims.nbDims; k++ ) {
                std::cout << dims.d[ k ];
                if ( k < dims.nbDims - 1 )
                    std::cout << "x";
            }
            std::cout << "] ";
        }
        std::cout << std::endl;

        // 打印输出维度
        std::cout << "  Outputs: ";
        for ( int j = 0; j < layer->getNbOutputs(); j++ ) {
            auto dims = layer->getOutput( j )->getDimensions();
            std::cout << "[";
            for ( int k = 0; k < dims.nbDims; k++ ) {
                std::cout << dims.d[ k ];
                if ( k < dims.nbDims - 1 )
                    std::cout << "x";
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }
    std::cout << "=====================\n\n\n" << std::endl;

    // 创建engine
    std::unique_ptr<IBuilderConfig> config( builder->createBuilderConfig() );
    std::unique_ptr<IHostMemory>    serializedEngine( builder->buildSerializedNetwork( *network, *config ) );

    // 创建runtime和engine
    std::unique_ptr<IRuntime> runtime( createInferRuntime( gLogger ) );
    ICudaEngine*              engine  = runtime->deserializeCudaEngine( serializedEngine->data(), serializedEngine->size() );
    IExecutionContext*        context = engine->createExecutionContext();

    // 准备输入数据
    float* inputData = new float[ batchSize * numPoints * channels ];
    generateRandomPointCloud( inputData, batchSize, numPoints, channels );

    // 分配GPU内存
    void* deviceInput;
    void* deviceOutput;
    cudaMalloc( &deviceInput, batchSize * numPoints * channels * sizeof( float ) );
    cudaMalloc( &deviceOutput, batchSize * numSamples * sizeof( int64_t ) );

    // 拷贝输入数据到GPU
    cudaMemcpy( deviceInput, inputData, batchSize * numPoints * channels * sizeof( float ), cudaMemcpyHostToDevice );

    // 执行推理
    void* bindings[] = { deviceInput, deviceOutput };
    context->executeV2( bindings );

    // 获取结果
    int64_t* outputData = new int64_t[ batchSize * numSamples ];
    cudaMemcpy( outputData, deviceOutput, batchSize * numSamples * sizeof( int64_t ), cudaMemcpyDeviceToHost );

    // 打印部分结果
    std::cout << "fps sample result: " << std::endl;
    for ( int i = 0; i < 10; i++ ) {
        std::cout << outputData[ i ] << " ";
    }
    std::cout << std::endl;

    // 清理资源
    delete[] inputData;
    delete[] outputData;
    cudaFree( deviceInput );
    cudaFree( deviceOutput );
    delete context;
    delete engine;
    delete plugin;
    delete creator;

    // 在创建插件后添加
    // std::cout << "Plugin type name: " << plugin-> << std::endl;      // 会输出
    // "KDTreeFpsample"
    std::cout << "Plugin instance name: " << layer->getName() << std::endl;  // 会输出 "fpsample"

    return 0;
}