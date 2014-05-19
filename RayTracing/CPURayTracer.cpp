/*
 *	FileName:	CPURayTracer.cpp
 *
 *	Programmer:	Jiayin Cao
 */

#include "define.h"
#include "CPURayTracer.h"
#include "D3DResource.h"

//constructor and destructor
CPURayTracer::CPURayTracer()
{
	InitializeDefault();
}

CPURayTracer::~CPURayTracer()
{
	SAFE_DELETEARRAY(m_pImageBackBuffer);
}

//initialize default value
void CPURayTracer::InitializeDefault()
{
	m_ElapsedTime = 0;
	m_iImageWidth = 0;
	m_iImageHeight = 0;
	m_pImageBuffer = 0;
	m_pScene = 0;
	m_pKDTreeBuffer = 0;
	m_pVertexBuffer = 0;
	m_pIndexBuffer = 0;
	m_pAttributeBuffer = 0;
	m_pMaterialBuffer = 0;
	m_pCustomTexture = 0;
	m_pTextureOffset = 0;
	m_pNormalBuffer = 0;
	m_pTexCoordinateBuffer = 0;
	m_bForceStop = true;
	m_pImageBackBuffer = 0;
}

//set image resolution
void CPURayTracer::SetImageResolution( int w , int h )
{
	//check if there is need to allocate the data for new resolution
	if( w == m_iImageWidth && h == m_iImageHeight )
		return;

	//copy the data
	m_iImageWidth = w;
	m_iImageHeight = h;

	float aspect = (float) w / (float) h;

	D3DXMatrixPerspectiveFovLH( &m_ProjectionMatrix , D3DX_PI/4 , aspect , 1.0f , 5000.0f );

	//allocate the memory
	SAFE_DELETEARRAY(m_pImageBackBuffer);
	m_pImageBackBuffer = new _float4[ w * h ];
}

//set projection and view matrix
void CPURayTracer::SetMatrix( D3DXMATRIX* view )
{
	D3DXMatrixInverse( &m_InvViewMatrix , NULL , view );

	m_EyePosition[0] = m_InvViewMatrix._41;
	m_EyePosition[1] = m_InvViewMatrix._42;
	m_EyePosition[2] = m_InvViewMatrix._43;
}

//set buffer
void CPURayTracer::SetImageBuffer( COLORREF* buf )
{
	m_pImageBuffer = buf;
}

//set scene info
void CPURayTracer::SetScene( CustomScene* scene )
{
	m_pScene = scene;
}

//bind render target
void CPURayTracer::BindRenderTarget( COLORREF* buf , int w , int h )
{
	//set the buffer first
	SetImageBuffer( buf );

	//set image resolution
	SetImageResolution( w , h );
}

//bind buffer for current scene
void CPURayTracer::BindBuffer()
{
	//set the buffer first
	m_pKDTreeBuffer = m_pScene->GetKDTree()->GetBuffer();
	m_pVertexBuffer = m_pScene->GetVertexBuffer();
	m_pIndexBuffer = m_pScene->GetKDTree()->GetIndexBuffer();
	m_pMaterialBuffer = m_pScene->GetMaterialBuffer();
	m_pAttributeBuffer = m_pScene->GetAttributeBuffer();
	m_pCustomTexture = D3DResource::GetSingleton()->GetCustomTexture();
	m_pTextureOffset = D3DResource::GetSingleton()->GetTextureOffset();
	m_pNormalBuffer = m_pScene->GetNormalBuffer();
	m_pTexCoordinateBuffer = m_pScene->GetTextureCoodinateBuffer();
	m_pOffsetBuffer = m_pScene->GetKDTree()->GetOffsetBuffer();
}

//ray tracing
void CPURayTracer::RayTrace()
{
	//reset the timer
	m_Timer.Reset();
	//start the timer
	m_Timer.Start();

	//bind the buffer
	BindBuffer();

	for( int j = m_iImageHeight - 1 ; j > -1 ; j-- )
	{
		for( int i = 0 ; i < m_iImageWidth; i++ )
		{
			//current index
			int	currentIndex = j * m_iImageWidth + i;

			//current ray
			_float4 dir , ori;

			//generate a ray for current pixel
			GenerateRay( i , j , &ori , &dir );

			//copy the image
			m_pImageBackBuffer[ currentIndex ] = Trace( ori , dir );

			//copy the image
			m_pImageBuffer[ currentIndex ] = RGB_FLOAT4( m_pImageBackBuffer[ currentIndex ] );

			//update current pixel number
			m_iCurrentPixelNum++;

			if( m_bForceStop )
				break;
		}

		if( m_bForceStop )
			break;
	}

	//clear the noise
	ClearNoise( m_pImageBackBuffer );

	//end the timer
	m_Timer.Stop();
	m_ElapsedTime = (int)m_Timer.GetElapsedTime();
}

//trace a ray
_float4 CPURayTracer::Trace( _float4 ori , _float4 dir )
{
	//current color
	_float4 d( 0 , 0 , 0 , 0 );

	//traverse a ray
	if( TraverseRay( ori , dir , &d ) )
	{
		//the final color
		_float4 color = PixelShader( &d , ori );

		//shade the pixel
		return color;
	}

	//return back ground color
	return d;
}

//Generate a ray for current pixel
void CPURayTracer::GenerateRay( int x , int y , _float4* ori , _float4* dir )
{
	D3DXVECTOR3	v;
	v.x = ( ( ( 2.0f * x ) / m_iImageWidth ) - 1 ) / m_ProjectionMatrix._11;
	v.y = -1.0f * ( ( ( 2.0f * ( m_iImageHeight - y - 1 ) ) / m_iImageHeight ) - 1 ) / m_ProjectionMatrix._22;
	v.z = 1.0f;

	//set the direction of the ray
	if( dir )
	{
		(*dir)[0] = v.x * m_InvViewMatrix._11 + v.y * m_InvViewMatrix._21 + v.z * m_InvViewMatrix._31;
		(*dir)[1] = v.x * m_InvViewMatrix._12 + v.y * m_InvViewMatrix._22 + v.z * m_InvViewMatrix._32;
		(*dir)[2] = v.x * m_InvViewMatrix._13 + v.y * m_InvViewMatrix._23 + v.z * m_InvViewMatrix._33;
	}

	//set the eye position
	if( ori )
	{
		(*ori)[0] = m_InvViewMatrix._41;
		(*ori)[1] = m_InvViewMatrix._42;
		(*ori)[2] = m_InvViewMatrix._43;
	}

	normalize(*dir);
}

//intersection test for a ray and a bounding box
bool CPURayTracer::GetIntersectedBoundingBox( _float4 ori , _float4 dir , _float4 min , _float4 max , _float4* intersected , float len )
{
	//set default value for tMin and tMax
	float tMin = 0 , tMax = FLT_MAX;

	if( len > 0 )
		tMax = len;

	for( int axis = 0 ; axis < 3 ; axis++ )
	{
		//if the ray is parrallel with the slab , check it seperately
		if( fabs( dir[axis] ) < 0.000001f )
		{
			if( ori[axis] > max[axis] || ori[axis] < min[axis] )
				return false;
		}else
		{
			//compute intersection t value of the ray with near and far plane of the slab
			float ood = 1.0f / dir[axis];
			float t1 = ( max[axis] - ori[axis] ) * ood;
			float t2 = ( min[axis] - ori[axis] ) * ood;

			//swap the two values
			if( t1 > t2 )
			{
				float temp = t1;
				t1 = t2;
				t2 = temp;
			}

			//clamp the tmin and tmax
			tMin = max( t1 , tMin );
			tMax = min( t2 , tMax );

			if( tMin > tMax )
				return false;
		}
	}

	//set intersected point
	if( intersected )
		*intersected = ori + dir * tMin;

	return true;
}

//get intersected point between a ray and a plane
bool CPURayTracer::GetIntersectedPlane( _float4 ori , _float4 dir , _float4 v1 , _float4 v2 , _float4 v3 , _float4* intersected )
{
	//get the edge
	_float4 e1 , e2 , normal;

	e1 = v2 - v1;
	e2 = v3 - v1;
	normal = cross( e1 , e2 );

	if( fabs( dot( normal , dir ) ) < 0.000001f )
		return false;

	float t = dot( normal , ori - v1 ) / dot( normal , dir );

	if( intersected ) 
	{
		*intersected = ori - t * dir;
		intersected->w = -t;
	}

	return t <= 0 ;
}

//get intersected point between a ray and a triangle
bool CPURayTracer::GetIntersectedTriangle( _float4 ori , _float4 dir , _float4 v1 , _float4 v2 , _float4 v3 , _float4* intersected )
{
	//check if the ray intersect with current plane
	bool inter = GetIntersectedPlane( ori , dir , v1 , v2 , v3 , intersected );

	if( inter == false )
		return false;

	//get the edge of the triangle
	_float4 e1 , e2 , e3;

	e1 = v1 - v2;
	e2 = v2 - v3;
	e3 = v3 - v1;

	_float4 rv1 , rv2 , rv3;
	rv1 = *intersected - v2 ;
	rv2 = *intersected - v3 ;
	rv3 = *intersected - v1 ;

	_float4 d1 , d2 , d3;
	d1 = cross( rv1 , e1 );
	d2 = cross( rv2 , e2 );
	d3 = cross( rv3 , e3 );

	//the dot product
	float f1 = dot( d1 , d2 );
	float f2 = dot( d2 , d3 );

	if( !( f1 >= -0.00000000000001f && f2 >= -0.00000000000001f ) )
		return false;

	return true;
}

//check if a point is in the bounding box
bool CPURayTracer::PointInBoundingBox( _float4 p , _float4 min , _float4 max )
{
	float threshold = 0.001f;

	if( p.x < min.x - threshold || p.y < min.y - threshold || p.z < min.z - threshold ||
		p.x > max.x + threshold || p.y > max.y + threshold || p.z > max.z + threshold )
		return false;

	return true;
}

//traverse a ray through kd-tree to find a intersected point
bool CPURayTracer::TraverseRay( _float4 ori , _float4 dir , _float4* intersected , int nodeIndex , float len )
{
	//the inPoint
	_float4	inPoint;

	//get the pointer of the node
	float*	node = m_pKDTreeBuffer + nodeIndex * 16;

	//find the intersection point with bounding box of kd-tree root
	bool cross = GetIntersectedBoundingBox( ori , dir , node + 8 , node + 12 , &inPoint , len );

	if( cross == false )
		return false;

	//check if the ray is a leaf node
	if( node[4] < 0 )
	{
		//get the triangle number
		int triNum = (int)node[3];
		//get the offset in index buffer
		int offset = m_pOffsetBuffer[nodeIndex];

		cross = false;
		float t = FLT_MAX;
		_float4 result;
		int id = -1;

		if( len > 0 )
			t = len;

		//iterate all of the triangles in the node
		for( int i = 0 ; i < triNum ; i++ )
		{
			//get the triangle index in the current buffer
			int index = m_pIndexBuffer[offset + i];

			//get the triangle pointer
			_float4* v1 = m_pVertexBuffer + index * 3;
			_float4* v2 = v1 + 1;
			_float4* v3 = v2 + 1;

			//get the intersected point of the ray and the triangle
			if( GetIntersectedTriangle( ori , dir , *v1 , *v2 , *v3 , intersected ) )
			{
				if( PointInBoundingBox( *intersected , node + 8 , node + 12 ) && intersected->w < t )
				{
					result = *intersected;
					t = intersected->w;
					cross = true;
					result.w = (float)index;

					if( len > 0 )
						break;
				}
			}
		}
		
		//copy the result
		*intersected = result;

		return cross;

	}else
	{
		//get the split plane
		int		splitAxis = (int) node[4];
		float	splitPos = node[5];
		
		//the flag
		int	flag = ( inPoint[splitAxis] < splitPos )? 0 : 1;

		//get to the child node
		int childIndex = (int)node[ 1 + flag ];

		//traverse the ray to the child node
		cross = TraverseRay( ori , dir , intersected , childIndex , len );

		if( cross == false )
		{
			childIndex = (int)node[ 2 - flag ];
			cross = TraverseRay( ori , dir , intersected , childIndex , len );
		}
	}

	return cross;
}

//shade a pixel
_float4 CPURayTracer::PixelShader( _float4* intersected , _float4 ori , float den )
{
	if( den < 0.01f )
		return _float4( 0 , 0 , 0 , 0 );

	//get triangle index
	int index = (int) intersected->w;

	//get the vertex
	_float4* v1 = (_float4*)(m_pVertexBuffer + 3 * index );
	_float4* v2 = v1 + 1;
	_float4* v3 = v2 + 1;

	//get the normal
	_float4* n1 = m_pNormalBuffer + 3 * index;
	_float4* n2 = n1 + 1;
	_float4* n3 = n2 + 1;

	//interplote normals
	_float4 factor , normal;
	GetInterploteFactor( *v1 , *v2 , *v3 , *intersected , &factor );
	GetInterplotedNormal( factor , *n1 , *n2 , *n3 , &normal );

	//get the reflect direction
	_float4 eyeDir , reflectDir ;

	eyeDir = *intersected - ori;
	normalize( eyeDir );

	//get the material
	UINT matIndex = m_pAttributeBuffer[index];
	Custom_Material* mat = m_pMaterialBuffer[matIndex];

	_float4 color = mat->m_Ambient;

	for( int i = 0 ; i < m_pScene->GetLightNumber() ; i++ )
	{
		_float4 lightPos;
		m_pScene->GetLightPosition( (float*) &lightPos , i );

		_float4 lightDir = *intersected - lightPos;
		normalize( lightDir );

		//get the dot product of normal and light dir
		float density = clamp( -dot( normal , lightDir )*lightPos.w);

		if( density < 0.01f )
			continue;

		//enable shadow
		if( PointInShadow( *intersected , lightPos ) )
			continue;

		if( mat->m_iTextureIndex < 0 )
		{
			color = color + mat->m_Diffuse * density ;
		}else
		{
			//get the normal
			float* t1 = m_pTexCoordinateBuffer + 6 * index;
			float* t2 = t1 + 2;
			float* t3 = t2 + 2;

			//get texture coordinate
			_float4 texDif;
			float tex[2];
			GetInterplotedTextureCoord( (float*)&factor , t1 , t2 , t3 , tex );

			//get the texture
			GetPixelFromTexture( mat->m_iTextureIndex , tex[0] , tex[1] , &texDif );

			//update color
			color = color + texDif * mat->m_Diffuse * density;
		}

		//add specular
		if( mat->m_nPower > 0 )
		{
			//reflect direction
			reflectDir = reflect( lightDir , normal );
			normalize( reflectDir );

			//get the dot product
			float d = clamp(-dot( reflectDir , eyeDir ));

			if( d > 0 )
				color = color + pow( d , mat->m_nPower ) * mat->m_Specular;
		}
	}

	color = color * den;

	//check if there is reflection
	if( fabs( dot( eyeDir , normal ) ) > 0.01 )
	{
		if( mat->m_fReflect * den > 0.1f  )
		{
			//get the reflection
			reflectDir = reflect( eyeDir , normal );
			normalize( reflectDir );

			//get the position
			_float4 reflectOri = reflectDir * 0.1f + *intersected;
		
			_float4 inter;
			//traverse a ray
			if( TraverseRay( reflectOri , reflectDir , &inter ) )
			{
				//shade the pixel
				color = color + PixelShader( &inter , reflectOri , mat->m_fReflect * den );
			}
		}else if( mat->m_fRefract * den > 0.1f )
		{
			//get the reflect direction
			_float4 refractDir = refract( eyeDir , normal , 1.0f / mat->m_fRefractRate );

			//get the position
			_float4 refractOri = refractDir * 0.1f + *intersected;
			normalize( refractDir );

			_float4 inter;
			//traverse a ray
			if( TraverseRay( refractOri , refractDir , &inter ) )
			{
				//shade the pixel
				color = color + PixelShader( &inter , refractOri , mat->m_fRefract * den );
			}
		}
	}

	saturate( color );

	return color;
}

//interplote normal
void CPURayTracer::GetInterploteFactor( _float4 v1 , _float4 v2 , _float4 v3 , _float4 intersected , _float4* factor )
{
	_float4 e1 , e2 , e3;

	e1 = intersected - v1;
	e2 = intersected - v2;
	e3 = intersected - v3;

	//the area
	_float4 c1 , c2 , c3;
	c1 = cross( e2 , e3 );
	c2 = cross( e3 , e1 );
	c3 = cross( e1 , e2 );

	factor->x = length( c1 );
	factor->y = length( c2 );
	factor->z = length( c3 );

	float s = factor->x + factor->y + factor->z;
	if( s != 0 )
	{
		for( int i = 0 ; i < 3 ; i++ )
			(*factor)[i] /= s;
	}
}

//get interploted normal
void CPURayTracer::GetInterplotedNormal( _float4 factor , _float4 n1 , _float4 n2 , _float4 n3 , _float4* result )
{
	*result = factor.x * n1 + factor.y * n2 + factor.z * n3;
}

//get interploted texture coordinate
void CPURayTracer::GetInterplotedTextureCoord( float* factor , float* t1 , float* t2 , float* t3 , float* result )
{
	for( int i = 0 ; i < 2 ; i++ )
		result[i] = factor[0] * t1[i] + factor[1] * t2[i] + factor[2] * t3[i];
}

//get a pixel from texture
void CPURayTracer::GetPixelFromTexture( int texIndex , float u , float v , _float4* color )
{
	//get the texture offset first
	UINT offset = m_pTextureOffset[texIndex];

	//get the data
	_float4* pData = m_pCustomTexture + offset;

	//clamp u , v first
	u -= floor(u);
	v -= floor(v);

	if( u < 0.0f )
		u += 1.0f;
	if( v < 0.0f )
		v += 1.0f;

	//get the index
	UINT x = (UINT)(pData[0].y * u);
	UINT y = (UINT)(pData[0].z * v);

	//return the result
	offset = (int)pData[0].y * y + x + 1;

	*color = *(pData + offset);
}

//check if the intersected point is in the shadow
bool CPURayTracer::PointInShadow( _float4 p , _float4 lightPos )
{
	//get the direction
	_float4 dir = p - lightPos;

	//get the length of the vector
	float len = length( dir ) * 0.98f;

	//normalize the vector
	normalize( dir );

	//check if there is triangle intersect with current segemeted line
	_float4 intersect;
	if( TraverseRay( lightPos , dir , &intersect , 0 , len ) )
		return true;

	return false;
}

//reset current pixel number
void CPURayTracer::ResetCurrentPixelNum()
{
	m_iCurrentPixelNum = 0;
}

//get current pixel number
int	CPURayTracer::GetCurrentPixelNum()
{
	return m_iCurrentPixelNum;
}

//get elapsed time
int	CPURayTracer::GetElapsedTime()
{
	return m_ElapsedTime;
}

//enable force stop
void CPURayTracer::ForceStop( bool enable )
{
	m_bForceStop = enable;
}

//clear the noise of the image
void CPURayTracer::ClearNoise( _float4* buffer )
{
	//copy the buffer
	for( int j = m_iImageHeight - 1 ; j > -1 ; j-- )
	{
		for( int i = 0 ; i < m_iImageWidth; i++ )
		{
			//current index
			int	currentIndex = j * m_iImageWidth + i;
			int leftIndex = j * m_iImageWidth - 1 + i;
			int rightIndex = j * m_iImageWidth + 1 + i;
			int upIndex = (j-1) * m_iImageWidth + i;
			int downIndex = (j+1) * m_iImageWidth + i;

			int difference = 0;
			float t = 0.4f;
			_float4 sum = _float4( 0 , 0 , 0 , 0 );
			if( i > 0 )
			{
				if( length( m_pImageBackBuffer[ currentIndex ] - m_pImageBackBuffer[leftIndex] ) > t )
					difference++;
				sum += m_pImageBackBuffer[leftIndex];
			}
			if( i < m_iImageWidth - 1 )
			{
				if( length( m_pImageBackBuffer[ currentIndex ] - m_pImageBackBuffer[rightIndex] ) > t )
					difference++;
				sum += m_pImageBackBuffer[rightIndex];
			}
			if( j > 0 )
			{
				if( length( m_pImageBackBuffer[ currentIndex ] - m_pImageBackBuffer[upIndex] ) > t )
					difference++;
				sum += m_pImageBackBuffer[upIndex];
			}
			if( j < m_iImageHeight - 1 )
			{
				if( length( m_pImageBackBuffer[ currentIndex ] - m_pImageBackBuffer[downIndex] ) > t )
					difference++;
				sum += m_pImageBackBuffer[downIndex];
			}

			_float4 color = m_pImageBackBuffer[ currentIndex ];
			if( difference >= 2 )
				color = sum * 0.25f;

			//copy the image
			m_pImageBuffer[ currentIndex ] = RGB_FLOAT4( color );

			if( m_bForceStop )
				break;
		}

		if( m_bForceStop )
			break;
	}
}