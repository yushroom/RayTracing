/*
 *	FileName:	RayTracer_Kernel.cu
 *
 *	Programmer:	Jiayin Cao
 */

//the sum for scan
int*	g_ScanSum[2];

//some helper functions
__device__ void d_normalize( float4* v )
{
	float s = v->x * v->x + v->y * v->y + v->z * v->z;
	s = sqrt(s);

	v->x /= s;
	v->y /= s;
	v->z /= s;
}

//cross product
__device__ float4 d_cross( const float4& v1 , const float4& v2 )
{
	float4 r;

	r.x = v1.y * v2.z - v1.z * v2.y;
	r.y = v1.z * v2.x - v1.x * v2.z;
	r.z = v1.x * v2.y - v1.y * v2.x;
	r.w = 0.0f;

	return r;
}

//clamp the value
__device__ float d_clamp( const float v )
{
	if( v > 1.0f )
		return 1.0f;
	if( v < 0.0f )
		return 0.0f;

	return v;
}

//clamp the float4
__device__ float4 d_saturate( const float4& v )
{
	return make_float4( d_clamp( v.x ) , d_clamp( v.y ) , d_clamp( v.z ) , d_clamp( v.w ) );
}

//dot product
__device__ float d_dot( const float4& v1 , const float4& v2 )
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z ;
}

//the length of the vector
__device__ float d_length( const float4& v )
{
	return sqrt( v.x * v.x + v.y * v.y + v.z * v.z );
}

//define some useful operators for float4
__device__ float4 operator+ ( const float4& v1 , const float4& v2 )
{
	return make_float4( v1.x + v2.x , v1.y + v2.y , v1.z + v2.z , v1.w + v2.w );
}

__device__ float4 operator- ( const float4& v1 , const float4& v2 )
{
	return make_float4( v1.x - v2.x , v1.y - v2.y , v1.z - v2.z , v1.w - v2.w );
}

__device__ float4 operator* ( const float4& v , const float d )
{
	return make_float4( v.x * d , v.y * d , v.z * d , v.w * d );
}

__device__ float4 operator* ( const float d , const float4& v )
{
	return make_float4( v.x * d , v.y * d , v.z * d , v.w * d ); 
}

__device__ float4 operator* ( const float4& v1 , const float4& v2 )
{
	return make_float4( v1.x * v2.x , v1.y * v2.y , v1.z * v2.z , v1.w * v2.w );
}

__device__ float4 operator+= ( float4& v1 , const float4& v2 )
{
	v1 = v1 + v2;

	return v1;
}

__device__ float2 operator * ( const float d , const float2& v )
{
	return make_float2( d * v.x , d * v.y );
}

__device__ float2 operator + ( const float2& v1 , const float2& v2 )
{
	return make_float2( v1.x + v2.x , v1.y + v2.y );
}

__device__ float2 operator - ( const float2& v1 , const float2& v2 )
{
	return make_float2( v1.x - v2.x , v1.y - v2.y );
}

__device__ float2 floor( const float2& v )
{
	int x = (int) v.x ;
	int y = (int) v.y ;
	return make_float2( x , y );
}

//reflect direction
__device__ float4 d_reflect( const float4& dir , const float4& normal )
{
	float dotProduct = ( -2.0f ) * d_dot( dir , normal );

	float4 r = dir + dotProduct * normal;

	return make_float4( r.x , r.y , r.z , 0.0f );
}

//refraction direction
__device__ float4	d_refract( const float4& dir , float4 normal , float rate )
{
	float4 r;

	if( d_dot( dir , normal ) > 0 )
	{
		normal = -1.0f * normal;
		rate = 1.0f / rate;
	}

	float cos = -1.0f * d_dot( dir , normal );
	float t = 1 - rate * rate * ( 1 - cos * cos );

	if( t < 0 )
	{
		r = d_reflect( dir , normal );
	}else
	{
		float cos2 = sqrt( t );
		r = rate * dir + ( rate * cos - cos2 ) * normal ;
	}

	return r;
}


//check if the ray intersects with bounding box
__device__ float4 kernelIntersectBoundingBox( float4& ori , float4& dir , float4& min , float4& max , float length )
{
	//the result
	float4 result = make_float4( 0.0f , 9999999.0f , 0.0f , 0.0f );

	//limit the maxium value
	if( length > 0 )
		result.y = length;

	//the variables
	float t1 , t2;

	if( fabs( dir.x ) < 0.0000001f )
	{
		if( ori.x > max.x || ori.x < min.x )
			return result;
	}else
	{
		t1 = ( max.x - ori.x ) / dir.x;
		t2 = ( min.x - ori.x ) / dir.x;

		if( t1 > t2 ) { float t = t1; t1 = t2; t2 = t; }

		//clamp
		if( t1 > result.x ) result.x = t1;
		if( t2 < result.y ) result.y = t2;

		if( result.x > result.y )
			return result;
	}

	if( fabs( dir.y ) < 0.0000001f )
	{
		if( ori.y > max.y || ori.y < min.y )
			return result;
	}else
	{
		t1 = ( max.y - ori.y ) / dir.y;
		t2 = ( min.y - ori.y ) / dir.y;

		if( t1 > t2 ) { float t = t1; t1 = t2; t2 = t; }

		//clamp
		if( t1 > result.x ) result.x = t1;
		if( t2 < result.y ) result.y = t2;

		if( result.x > result.y )
			return result;
	}

	if( fabs( dir.y ) < 0.0000001f )
	{
		if( ori.z > max.z || ori.z < min.z )
			return result;
	}else
	{
		t1 = ( max.z - ori.z ) / dir.z;
		t2 = ( min.z - ori.z ) / dir.z;

		if( t1 > t2 ) { float t = t1; t1 = t2; t2 = t; }

		//clamp
		if( t1 > result.x ) result.x = t1;
		if( t2 < result.y ) result.y = t2;

		if( result.x > result.y )
			return result;
	}

	//enable the intersected point
	result.z = 1.0f;

	return result;
}

//check if the ray intersects with a plane
__device__ float4 kernelIntersectPlane( const float4& v1 , const float4& v2 , const float4& v3 , const float4& ori , const float4& dir )
{
	//w : >= 0 ( intersected point enable ) , < 0 ( disable )
	float4 result = make_float4( 0.0f , 0.0f , 0.0f , 0.0f );
	
	//get the normal of the plane
	float4 normal = d_cross( v2 - v1 , v3 - v1 );

	//get the factor
	float t = d_dot( normal , ori - v1 ) / d_dot( normal , dir );

	//set the result
	result = ori - t * dir;

	if( t <= 0.0f )
		result.w = -t;
	else
		result.w = -1;

	return result;
}

//check if the ray intersects with a triangle
__device__ float4 kernelIntersectTriangle( const float4& v1 , const float4& v2 , const float4& v3 , const float4& ori , const float4& dir )
{
	//the result
	float4 result = kernelIntersectPlane( v1 , v2 , v3 , ori , dir );

	if( result.w < 0 )
		return result;

	//get the factor
	float4 d1 = d_cross( result - v2 , v1 - v2 );
	float4 d2 = d_cross( result - v3 , v2 - v3 );
	float4 d3 = d_cross( result - v1 , v3 - v1 );

	float f1 = d_dot( d1 , d2 );
	float f2 = d_dot( d2 , d3 );

	if( !( f1 >= -0.000000000000001f && f2 >= -0.000000000000001f ) )
		result.w = -1.0f;

	return result;
}

//check if the current point is in the bounding box
__device__ int kernelPointInBoundingBox( const float4& p , const float4& min , const float4& max )
{
	float threshold = 0.00001f;

	if( p.x < min.x - threshold || p.y < min.y - threshold || p.z < min.z - threshold ||
		p.x > max.x + threshold || p.y > max.y + threshold || p.z > max.z + threshold )
		return false;

	return true;
}

//do interplotation
__device__ float4 kernelInterploted( const float4& v1 , const float4& v2 , const float4& v3 , const float4& intersected )
{
	//get the vectors
	float4 e1 = intersected - v1;
	float4 e2 = intersected - v2;
	float4 e3 = intersected - v3;

	//compute the areas
	float4 area;
	area.x = d_length( d_cross( e2 , e3 ) );
	area.y = d_length( d_cross( e3 , e1 ) );
	area.z = d_length( d_cross( e1 , e2 ) );

	float d = 1.0f / ( area.x + area.y + area.z );

	return area * d;
}

//clear and initialize buffer
__global__ void kernelInitBuffer(	float4* buffer , 
									int*	markedBuffer , 
									int		pixelNum )
{
	//get the thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//limit the thread id
	if( tid >= pixelNum )
		return;

	buffer[tid] = make_float4( 0.0f , 0.0f , 0.0f , 0.0f );
	markedBuffer[tid] = tid;
}

//generate primary ray intersected result
__global__ void kernelGenerateIntersectedPoint( float4* rayOri , 
												float4* rayDir ,
												float4*	vertexBuffer ,
												int		rayNum ,
												int*	index , 
												float4*	result )
{
	//get the thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//limit the thread id
	if( tid >= rayNum )
		return;

	//Load the vertex
	int triId = index[tid];

	//get the vertex
	int id = 3 * triId;
	float4 v1 = vertexBuffer[id];
	float4 v2 = vertexBuffer[id+1];
	float4 v3 = vertexBuffer[id+2];

	//ray ori and dir
	float4 ori = rayOri[tid];
	float4 dir = rayDir[tid];

	//get the intersected result
	result[tid] = kernelIntersectPlane( v1 , v2 , v3 , ori , dir );
	result[tid].w = triId;
}

//Generate primary rays
__global__ void kernelGeneratePrimaryRays(	float4	viewInfo ,	
											float*	invViewMatrix ,
											float4*	rayOri , 
											float4* rayDir )
{
	//get the thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//limit the thread id
	if( tid >= (int)viewInfo.x * (int)viewInfo.y )
		return;

	// get the pixel coorindate first
	uint2 coord;
	coord.x = tid % (int) viewInfo.x;
	coord.y = tid / (int)viewInfo.x;

	// compute the vector of the ray in screen space
	float2 v;
	v.x = ( ( ( 2.0f * coord.x ) / viewInfo.x ) - 1.0f ) / viewInfo.z;
	v.y = -1.0f * ( ( ( 2.0f * coord.y ) / viewInfo.y ) - 1.0f ) / viewInfo.w;

	//copy the original point of the rays
	rayOri[tid] = make_float4( invViewMatrix[12] , invViewMatrix[13] , invViewMatrix[14] , tid ); 

	//compute the direction of the ray
	float4 dir;
	dir.x = ( v.x * invViewMatrix[0] + v.y * invViewMatrix[4] + invViewMatrix[8] );
	dir.y = ( v.x * invViewMatrix[1] + v.y * invViewMatrix[5] + invViewMatrix[9] );
	dir.z = ( v.x * invViewMatrix[2] + v.y * invViewMatrix[6] + invViewMatrix[10] );
	dir.w = 0.0f;
	d_normalize( &dir );

	rayDir[tid] = make_float4( dir.x , dir.y , dir.z , 1.0f );
}

//traverse the ray through kd-tree
__device__ float4 kernelTraverseRay(	float4*	kdTree , 
										int*	indexMap , 
										int*	offsetBuffer ,
										float4*	vertexBuffer ,
										float4&	rayOri , 
										float4&	rayDir ,
										float	length )
{
	//the intersected result
	float4 result = make_float4( 0.0f , 0.0f , 0.0f , -1.0f );

	//tree node information
	float4 header;
	float4 splitInfo;

	//the bounding box
	float4 minBB = kdTree[2];
	float4 maxBB = kdTree[3];

	//check if the ray intersects with the current bounding box of the root
	result = kernelIntersectBoundingBox( rayOri , rayDir , minBB , maxBB , length );

	//if the ray doesn't cross the kd-tree , just return
	if( result.z < 0.5f )
	{
		result = make_float4( 0.0f , 0.0f , 0.0f , -1.0f );
		return result;
	}

	//current traversing node
	int currentNodeIndex = 0;

	//the mask to mark the traversed node
	unsigned int	mask = 0;
	//current traverse depth
	int				currentTraverseDepth = 0;

	//current inPonit when traversing the node
	float4 inPoint = rayOri + result.x * rayDir ;

	while( currentTraverseDepth >= 0 )
	{
		//traverse the current node
		do
		{
			//the current node offset
			int currentNodeOffset = currentNodeIndex * 4;

			//get the current node information
			header = kdTree[ currentNodeOffset ];
			splitInfo = kdTree[currentNodeOffset + 1 ];

			//check if it's a leaf node
			if( splitInfo.x < 0 )
				break;

			//get the split axis
			int splitAxis = (int) splitInfo.x;

			//get the pointer of the inPoint
			float sPos = 0.0f;

			if( splitAxis == 0 )
				sPos = inPoint.x;
			else if( splitAxis == 1 )
				sPos = inPoint.y;
			else if( splitAxis == 2 )
				sPos = inPoint.z;

			//update the virtual stack and traverse the node
			if( splitInfo.y > sPos )
				currentNodeIndex = (int)header.y;
			else
				currentNodeIndex = (int)header.z;

			//increase the current traverse depth
			currentTraverseDepth++;

		}while( true );

		//get the offset and triangle number
		int triOffset = offsetBuffer[currentNodeIndex];
		int triNumber = (int)header.w;

		//min value
		float minFactor = 9999999.0f;
		if( length > 0 )
			minFactor = length;

		//triangle index
		int oriTriIndex = -1;
		//the bounding box
		minBB = kdTree[currentNodeIndex*4+2];
		maxBB = kdTree[currentNodeIndex*4+3];
		//intersect with the current triangles
		for( int i = 0 ; i < triNumber ; i++ )
		{
			//get the triangles
			int triIndex = indexMap[triOffset+i];

			//get the vertex
			float4 v1 = vertexBuffer[3*triIndex];
			float4 v2 = vertexBuffer[3*triIndex+1];
			float4 v3 = vertexBuffer[3*triIndex+2];

			//get the intersected point
			result = kernelIntersectTriangle( v1 , v2 , v3 , rayOri , rayDir );

			//limit the factor
			if( result.w > 0.0f && result.w < minFactor )
			{
				if( kernelPointInBoundingBox( result , minBB , maxBB ) )
				{
					minFactor = result.w;
					oriTriIndex = triIndex;

					if( length > 0 )
						break;
				}
			}
		}

		if( oriTriIndex >= 0 )
		{
			result = rayOri + minFactor * rayDir;
			result.w = (float)oriTriIndex;
			return result;
		}

		//back track here
		while( currentTraverseDepth >= 0 )
		{
			if( currentTraverseDepth == 0 )
				return make_float4( 0 , 0 , 0 , -1.0f );

			//get the current mask
			if( mask & ( 0x00000001 << currentTraverseDepth ) )
			{
				//update the mask
				mask &= ~(0x00000001 << currentTraverseDepth );

				//decrease the current depth;
				currentTraverseDepth--;

				//get to the father node
				currentNodeIndex = (int)kdTree[ 4 * currentNodeIndex ].x;

				//continue to next level
				continue;
			}

			//check the other node
			int otherNode = currentNodeIndex + 1;
			if( currentNodeIndex % 2 == 0 )
				otherNode -= 2;

			//get the bounding box of the other node
			int otherNodeOffset = 4 * otherNode;
			minBB = kdTree[ otherNodeOffset + 2 ];
			maxBB = kdTree[ otherNodeOffset + 3 ];

			//get the intersected result
			float4 bi = kernelIntersectBoundingBox( rayOri , rayDir , minBB , maxBB , length );

			if( bi.z > 0.5f )
			{
				//update the current traverse node
				currentNodeIndex = otherNode;

				//update the inPoint
				inPoint = rayOri + bi.x * rayDir ;

				//update the mask
				mask |= 0x00000001 << currentTraverseDepth;

				break;
			}else
			{
				//update the mask
				mask &= ~( 0x00000001 << currentTraverseDepth );

				//decrease current depth
				currentTraverseDepth--;

				//get to the father node
				currentNodeIndex = (int) kdTree[ 4 * currentNodeIndex ].x;
			}
		}
	}

	result.w = -1.0f;
	return result;
}

//get the interseced point
__global__ void kernelGetIntersectedPoint(	float4*	rayOri , 
											float4*	rayDir ,
											float4*	kdTree ,
											int*	indexMap ,
											int*	offsetBuffer ,
											float4*	vertexBuffer ,
											int		rayNumber , 
											float4*	result )
{
	//get the thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//limit the thread id
	if( tid >= rayNumber )
		return;

	//get the triangle
	result[tid] = kernelTraverseRay( kdTree , indexMap , offsetBuffer , vertexBuffer , rayOri[tid] , rayDir[tid] , -1.0f );
}

//do pixel shader here
__global__ void kernelPixelShader(	float4*	intersected , 
									float4*	vertexBuffer , 
									float4*	normalBuffer ,
									float2*	texCoordinateBuffer , 
									float4*	kdTree ,
									int*	indexMap ,
									int*	offsetIndexBuffer,
									float4*	lightBuffer ,
									int*	attributeBuffer , 
									float4*	materialBuffer ,
									int*	textureOffset , 
									float4*	customTexture , 
									int		pixelNum , 
									float4*	rayDir ,
									int*	offsetBuffer ,
									float4*	destNormalBuffer , 
									float4*	imageBuffer )
{
	//get the thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//limit the thread id
	if( tid >= pixelNum )
		return;

	//get the triangle index
	int triIndex = (int)intersected[tid].w;
	int triOffset = 3 * triIndex;

	float4 color = make_float4( 0.0f , 0.0f , 0.0f , 0.0f );

	//load the density of the pixel
	if( triIndex < 0 )
		return;

	//get the material index
	int	matIndex = attributeBuffer[triIndex];

	//the material buffer
	float4 ambient = materialBuffer[ 4 * matIndex ];
	float4 diffuse = materialBuffer[ 4 * matIndex + 1 ];
	float4 specular = materialBuffer[ 4 * matIndex + 2 ];
	float4 matprop = materialBuffer[ 4 * matIndex + 3 ];
	
	//load the vertex
	float4 v1 = vertexBuffer[ triOffset ];
	float4 v2 = vertexBuffer[ triOffset + 1 ];
	float4 v3 = vertexBuffer[ triOffset + 2 ];

	//get the interploted
	float4 interploted = kernelInterploted( v1 , v2 , v3 , intersected[tid] );

	//get the normal
	float4 n1 = normalBuffer[ triOffset ];
	float4 n2 = normalBuffer[ triOffset + 1 ];
	float4 n3 = normalBuffer[ triOffset + 2 ];
	float4 normal = n1 * interploted.x + n2 * interploted.y + n3 * interploted.z;
	d_normalize( &normal );

	//update the normal buffer
	destNormalBuffer[tid] = normal;
	destNormalBuffer[tid].w = matIndex;

	//the density for the pixel
	float density = rayDir[tid].w;

	if( matprop.x > -0.5f )
	{
		//load the texture coordinate
		float2 t1 = texCoordinateBuffer[ triOffset ];
		float2 t2 = texCoordinateBuffer[ triOffset + 1 ];
		float2 t3 = texCoordinateBuffer[ triOffset + 2 ];
		float2 texCoord = interploted.x * t1 + interploted.y * t2 + interploted.z * t3;
		texCoord = texCoord - floor( texCoord );
		if( texCoord.x < 0.0f ) texCoord.x += 1.0f;
		if( texCoord.y < 0.0f ) texCoord.y += 1.0f;

		//load the texture
		float4* imgData = customTexture + textureOffset[(int)matprop.x];

		int x = imgData[0].y * texCoord.x ;
		int y = imgData[0].z * texCoord.y ;

		int texOffset = y * imgData[0].y + x + 1;

		diffuse = diffuse * (*(imgData + texOffset)) ;
	}

	//initialize the image buffer
	color = ambient;

	//shade the pixels
	for( int i = 0 ; i < 2 ; i++ )
	{
		if( lightBuffer[i].w < 0.01f )
			continue;

		//the light direction
		float4 lightDir = intersected[tid] - lightBuffer[i];

		//check if the point is in the shadow
		float shadowLen = 0.98f * d_length(lightDir);
		d_normalize( &lightDir );

		//the dot product
		float	dotProduct = d_dot( lightDir , normal );

		if( dotProduct > 0.0f )
			continue;

		{
			float4 shadowFactor = kernelTraverseRay( kdTree , indexMap , offsetIndexBuffer , vertexBuffer , lightBuffer[i] , lightDir , shadowLen );
			if( shadowFactor.w >= 0.0f )
				continue;
		}

		//the light density
		float lightDensity = d_clamp( -1.0f * dotProduct ) * lightBuffer[i].w;

		//load the density of current pixel
		color += diffuse * lightDensity ;

		//add specular if possible
		if( specular.w > 0 )
		{
			//reflect direction
			float4 reflectDir = d_reflect( lightDir , normal );
			d_normalize( &reflectDir );

			//get the dot product
			float d = d_clamp(-d_dot( reflectDir , rayDir[tid] ));

			if( d > 0 )
				color += pow( d , specular.w ) * specular;
		}
	}

	int offset = offsetBuffer[tid];
	imageBuffer[offset] = d_saturate( imageBuffer[offset] + d_saturate( color * density ) );
}

//generate next level rays
__global__ void kernelGenerateNextLevelRays(	float4* materialInfo , 
												float4*	intersected , 
												float4*	backNormalBuffer , 
												float4*	rayOri ,
												float4*	rayDir ,
												int		rayNumber ,
												float4*	destRayOri , 
												float4*	destRayDir ,
												int*	markedBuffer )
{
	//get the thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//limit the thread id
	if( tid >= rayNumber )
		return;

	//set marked buffer zero
	markedBuffer[tid] = 0;

	//load the intersected point
	float4 intersectedPoint = intersected[tid];

	//get the intersected triangle index
	int triIndex = (int)intersectedPoint.w;
	if( triIndex < 0 )
		return;

	//load the normal
	float4 normal = backNormalBuffer[tid];

	//get the material index
	int matIndex = (int)normal.w;

	//get the material
	float4 matInfo = materialInfo[4*matIndex+3];

	//load the ray direction
	float4 ori = rayOri[tid];
	float4 dir = rayDir[tid];

	//if there is reflection , mark result as true
	if( matInfo.y > 0 )
	{
		float4 reflectDir = d_reflect( dir , normal );
		d_normalize( &reflectDir );
		reflectDir.w = dir.w * matInfo.y;

		destRayDir[tid] = reflectDir;
		destRayOri[tid] = intersectedPoint + reflectDir * 0.1f;
		destRayOri[tid].w = ori.w;

		markedBuffer[tid] = 1;
	}else if( matInfo.z > 0 )
	{
		float4 refractDir = d_refract( dir , normal , 1.0f / matInfo.w );
		d_normalize( &refractDir );
		refractDir.w = dir.w * matInfo.z;

		destRayDir[tid] = refractDir;
		destRayOri[tid] = intersectedPoint + refractDir * 0.02f;
		destRayOri[tid].w = ori.w;

		markedBuffer[tid] = 1;
	}
}

//copy new rays
__global__ void kernelCopyNewRays(	float4* srcRayOri , 
									float4* srcRayDir ,
									int*	scanResult , 
									int		rayNumber , 
									float4* destRayOri ,
									float4* destRayDir ,
									int*	offsets )
{
	//get the thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//limit the thread id
	if( tid >= rayNumber )
		return;

	//load the offset
	int offset = scanResult[tid];

	if( offset != scanResult[tid+1] )
	{
		//set the result
		destRayOri[offset] = srcRayOri[tid];
		destRayDir[offset] = srcRayDir[tid];
		offsets[offset] = (int)srcRayOri[tid].w;
	}
}

//Do scan on GPU
__global__ void kernelScan( int* data , int number , int oBlockRes , int* blockRes )
{
	//the shared memory
	__shared__ int sharedMem[512];

	//get the thread id
	int ltid = threadIdx.x;
	int gtid = ltid + blockDim.x * blockIdx.x;

	//the block sum
	int blocksum = 0;

	//zero the rest of the memory
	if( 2 * gtid >= number )
	{
		data[ 2 * gtid ] = 0;
		data[ 2 * gtid + 1 ] = 0;
	}else if( 2 * gtid == number - 1 )
		data[ 2 * gtid + 1 ] = 0;

	//Load the data into the shared memory
	sharedMem[2*ltid] = data[2*gtid];
	sharedMem[2*ltid+1] = data[2*gtid+1];

	//the offset
	int offset = 1;

	for( int d = 256 ; d > 1 ; d >>= 1 )
	{
		//sync the threads in a group
		__syncthreads();

		if( ltid < d )
		{
			int ai = offset * ( 2 * ltid + 1 ) - 1;
			int bi = ai + offset;

			sharedMem[bi] += sharedMem[ai];
		}

		offset *= 2;
	}

	//the block sum
	blocksum = sharedMem[511] + sharedMem[255];

	//clear the last element
	if( ltid == 0 )
	{
		sharedMem[511] = sharedMem[255];
		sharedMem[255] = 0;
	}

	for( int d = 2 ; d < 512 ; d *= 2 )
	{
		__syncthreads();

		offset >>= 1;

		if( ltid < d )
		{
			int ai = offset * ( 2 * ltid + 1 ) - 1 ;
			int bi = ai + offset ;

			int t = sharedMem[ai];
			sharedMem[ai] = sharedMem[bi];
			sharedMem[bi] += t;
		}
	}

	__syncthreads();

	data[ 2 * gtid ] = sharedMem[ 2 * ltid ];
	data[ 2 * gtid + 1 ] = sharedMem[ 2 * ltid + 1 ];

	//Output Block Result
	if( oBlockRes > 0 )
	{
		if( ltid == 0 )
		{
			//copy the result
			blockRes[blockIdx.x] = blocksum;
		}
	}
}

//Add the block result to the segmented scan result
__global__ void kernelUniformAdd( int* data , int* blockResult )
{
	//get the thread id
	int ltid = threadIdx.x;
	int gtid = ltid + blockDim.x * blockIdx.x;
	
	//add the result
	data[gtid] += blockResult[gtid/512];
}

//clear the noise of the image
__global__ void kernelClearNoise(	float4* imgData , 
									int		width ,
									int		height ,
									float4* targetData )
{
	//get the thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//limit the thread id
	if( tid >= width * height )
		return;

	//threshold
	float threshold = 0.4f;

	//the difference
	int difference = 0;

	//current index
	int	currentIndex = tid;
	int leftIndex = tid - 1;
	int rightIndex = tid + 1;
	int upIndex = tid - width ;
	int downIndex = tid + width ;

	//the coordinate
	int i = tid % width;
	int j = tid / width;

	//current color
	float4 color = imgData[currentIndex];

	float4 sum = make_float4( 0 , 0 , 0 , 0 );
	if( i > 0 )
	{
		if( d_length( color - imgData[leftIndex] ) > threshold )
			difference++;
		sum += imgData[leftIndex];
	}
	if( i < width - 1 )
	{
		if( d_length( color - imgData[rightIndex] ) > threshold )
			difference++;
		sum += imgData[rightIndex];
	}
	if( j > 0 )
	{
		if( d_length( color - imgData[upIndex] ) > threshold )
			difference++;
		sum += imgData[upIndex];
	}
	if( j < height - 1 )
	{
		if( d_length( color - imgData[downIndex] ) > threshold )
			difference++;
		sum += imgData[downIndex];
	}

	if( difference >= 2 )
		color = sum * 0.25f;

	targetData[tid] = color;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//initialize buffer
extern "C" void cudaInitBuffer(	float4* buffer , 
								int*	markedBuffer , 
								int		pixelNum )
{
	//the block number
	int threadNum = 256;
	int blockNum = ( pixelNum + threadNum - 1 ) / threadNum;

	//call the kenrel
	kernelInitBuffer<<<blockNum,threadNum>>>( buffer , markedBuffer , pixelNum );
}

//generate primary ray intersected result
extern "C" void cudaGenerateIntersectedPoint(	float4* rayOri , 
												float4* rayDir ,
												float4*	vertexBuffer ,
												int		rayNum ,
												int*	index , 
												float4*	result )
{
	//the block number
	int threadNum = 256;
	int blockNum = ( rayNum + threadNum - 1 ) / threadNum;

	//call the kernel
	kernelGenerateIntersectedPoint<<<blockNum , threadNum>>>( rayOri , rayDir , vertexBuffer , rayNum , index , result );
}

//Generate primary rays
extern "C" void cudaGeneratePrimaryRays(	float4	viewInfo ,	
											float*	invViewMatrix ,
											float4*	rayOri , 
											float4* rayDir )
{
	//get the number of data
	int rayNum = (int)( viewInfo.x * viewInfo.y );

	//the block number
	int threadNum = 256;
	int blockNum = ( rayNum + threadNum - 1 ) / threadNum;

	//call the kernel
	kernelGeneratePrimaryRays<<<blockNum , threadNum>>>( viewInfo , invViewMatrix , rayOri , rayDir );
}

//get intersected point
extern "C" void cudaGetIntersectedPoint(	float4*	rayOri , 
											float4*	rayDir ,
											float4*	kdTree ,
											int*	indexMap ,
											int*	offsetBuffer ,
											float4*	vertexBuffer ,
											int		rayNumber , 
											float4*	result )
{
	//the block and thread number
	int threadNum = 256;
	int blockNum = ( rayNumber + threadNum - 1 ) / threadNum ;

	//call the kernel
	kernelGetIntersectedPoint<<<blockNum , threadNum>>>( rayOri , rayDir , kdTree , indexMap , offsetBuffer , vertexBuffer , rayNumber , result );
}

//do pixel shader
extern "C" void cudaPixelShader(	float4*	interseced , 
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
									float4*	imageBuffer )
{
	//the block and thread number
	int threadNum = 256;
	int blockNum = ( pixelNum + threadNum - 1 ) / threadNum ;

	//call the kernel
	kernelPixelShader<<<blockNum , threadNum>>>(	interseced , vertexBuffer , normalBuffer , texCoordinateBuffer , 
													kdTree , indexMap , offsetIndexBuffer , lightBuffer , attributeBuffer , materialBuffer , 
													textureOffset , customTexture , pixelNum , rayDir , offsetBuffer , destNormalBuffer , imageBuffer );
}

//generate next level rays
extern "C" void cudaGenerateNextLevelRays(	float4* materialInfo , 
											float4*	intersected , 
											float4*	backNormalBuffer , 
											float4*	rayOri ,
											float4*	rayDir ,
											int		rayNumber ,
											float4*	destRayOri , 
											float4*	destRayDir ,
											int*	markedBuffer )
{
	//the block and thread number
	int threadNum = 256;
	int blockNum = ( rayNumber + threadNum - 1 ) / threadNum ;

	//call the kernel
	kernelGenerateNextLevelRays<<<blockNum , threadNum>>>(	materialInfo , intersected , backNormalBuffer , rayOri , rayDir , 
															rayNumber , destRayOri , destRayDir , markedBuffer );
}

//do scan on gpu
extern "C" void cudaScan( int* data , int num , int level )
{
/*	//allocate the number of data
	int* cpuData = new int[num];

	//pass the data from gpu to cpu
	cudaMemcpy( cpuData , data , sizeof( int ) * ( num - 1 ) , cudaMemcpyDeviceToHost );

	int last = 0;
	for( int i = 0 ; i < num ; i++ )
	{
		int oldLast = last;
		last += cpuData[i];
		cpuData[i] = oldLast;
	}

	//pass the data back from cpu to gpu
	cudaMemcpy( data , cpuData , sizeof( int ) * num , cudaMemcpyHostToDevice );

	//delete the data
	delete[] cpuData;*/

	//the dimension of the kernel
	dim3 threads( 256 );
	dim3 blocks( ( num + 511 ) / 512 );

	//call the kernel
	kernelScan<<<blocks , threads>>>( data , num , 1 , g_ScanSum[level] );

	//scan the block Result
	if( num <= 262144 )
		kernelScan<<<1 , threads>>>( g_ScanSum[level] , blocks.x , -1 , data );
	else
		cudaScan( g_ScanSum[level] , blocks.x , level + 1 );

	//add the offset
	threads.x = 512;
	kernelUniformAdd<<< blocks , threads >>> ( data , g_ScanSum[level] );
}

//copy new rays
extern "C" void cudaCopyNewRays(	float4* srcRayOri , 
									float4* srcRayDir ,
									int*	scanResult , 
									int		rayNumber , 
									float4* destRayOri ,
									float4* destRayDir ,
									int*	offsets )
{
	//the block and thread number
	int threadNum = 256;
	int blockNum = ( rayNumber + threadNum - 1 ) / threadNum ;

	//call the kernel
	kernelCopyNewRays<<<blockNum , threadNum>>>( srcRayOri , srcRayDir , scanResult , rayNumber , destRayOri , destRayDir , offsets );
}

//clear the noise of the image
extern "C" void cudaClearNoise(	float4* imgData , 
								int		width ,
								int		height ,
								float4* targetData )
{
	//the block and thread number
	int threadNum = 256;
	int blockNum = ( width * height + 255 ) / 256;

	//call the kernel
	kernelClearNoise<<<blockNum , threadNum>>>( imgData , width , height , targetData );
}