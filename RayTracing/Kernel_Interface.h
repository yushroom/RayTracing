/*
 *	FileName:	Kernel_Interface.h
 *
 *	Programmer:	Jiayin Cao
 */

extern "C"
{
	//initialize buffer
	void cudaInitBuffer(	float4* buffer , 
							int*	markedBuffer , 
							int		pixelNum );

	//generate primary ray intersected result
	void cudaGenerateIntersectedPoint(	float4* rayOri , 
										float4* rayDir ,
										float4* vertexBuffer ,
										int		rayNum ,
										int*	index , 
										float4*	result );

	//Generate primary rays
	void cudaGeneratePrimaryRays(	float4	viewInfo ,	
									float*	invViewMatrix ,
									float4*	rayOri , 
									float4* rayDir );

	//get intersected point
	void cudaGetIntersectedPoint(	float4*	rayOri , 
									float4*	rayDir ,
									float4*	kdTree ,
									int*	indexMap ,
									int*	offsetBuffer ,
									float4*	vertexBuffer ,
									int		rayNumber , 
									float4*	result );

	//do pixel shader on cuda
	void cudaPixelShader(	float4*	interseced , 
							float4*	vertexBuffer , 
							float4*	normalBuffer ,
							float2*	texCoordinateBuffer , 
							float4*	kdTree ,
							int*	indexMap ,
							int*	offsetIndexBuffer ,
							float4*	lightBuffer ,
							int*	attributeBuffer , 
							float4*	materialBuffer ,
							int*	textureOffset , 
							float4*	customTexture ,
							int		pixelNum , 
							float4*	rayDir ,
							int*	offsetBuffer ,
							float4*	destNormalBuffer , 
							float4*	imageBuffer );

	//generate next level rays
	void cudaGenerateNextLevelRays(	float4* materialInfo , 
									float4*	intersected , 
									float4*	backNormalBuffer , 
									float4*	rayOri ,
									float4*	rayDir ,
									int		rayNumber ,
									float4*	destRayOri , 
									float4*	destRayDir ,
									int*	markedBuffer );

	//do scan on gpu
	void cudaScan( int* data , int num , int level = 0 );

	//copy new rays
	void cudaCopyNewRays(	float4* srcRayOri , 
							float4* srcRayDir ,
							int*	scanResult , 
							int		rayNumber , 
							float4* destRayOri ,
							float4* destRayDir ,
							int*	offsets );

	//clear the noise of the image
	void cudaClearNoise(	float4* imgData , 
							int		width ,
							int		height ,
							float4* targetData );
};