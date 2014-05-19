/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	CustomScene.cpp
 */

//include the headers
#include "CustomScene.h"
#include "define.h"
#include "D3DResource.h"

//constructor and destructor
CustomScene::CustomScene()
{
	InitializeDefault();
}

CustomScene::~CustomScene()
{
	//release the content
	ReleaseContent();
}

//initialize default
void CustomScene::InitializeDefault()
{
	m_pMeshList = NULL;
	m_pVertexBuffer = NULL;
	m_iVertexNumber = 0;
	m_pAttributeBuffer = 0;
	m_pMaterialBuffer = 0;
	m_pNormalBuffer = 0;
	m_pTexCoordinateBuffer = 0;

	m_cVertexBuffer = 0;
	m_cNormalBuffer = 0;
	m_cTexCoordinateBuffer = 0;
	m_cAttributeBuffer = 0;
	m_cMaterialBuffer = 0;
	m_cLightBuffer = 0;
}

//release the resource
void CustomScene::ReleaseContent()
{
	if( m_pMeshList == NULL )
		return;

	//release the content
	for( int i = 0 ; i < m_iEntityNumber ; i++ )
		m_pMeshList[i].ReleaseContent();

	//delete the array of mesh
	SAFE_DELETEARRAY(m_pMeshList);

	//delete the vertex buffer
	SAFE_DELETEARRAY(m_pVertexBuffer);
	SAFE_DELETEARRAY(m_pNormalBuffer);
	SAFE_DELETEARRAY(m_pTexCoordinateBuffer);

	//delete the attribute buffer
	SAFE_DELETEARRAY(m_pAttributeBuffer);
	SAFE_DELETEARRAY(m_pMaterialBuffer);

	//release the memory first
	SAFE_RELEASE_CUDA( m_cVertexBuffer );
	SAFE_RELEASE_CUDA( m_cNormalBuffer );
	SAFE_RELEASE_CUDA( m_cTexCoordinateBuffer );
	SAFE_RELEASE_CUDA( m_cAttributeBuffer );
	SAFE_RELEASE_CUDA( m_cMaterialBuffer );
	SAFE_RELEASE_CUDA( m_cLightBuffer );

	//release kd-tree content
	m_KDTree.ReleaseContent();
}

//Load scene from file
bool CustomScene::LoadScene( const char* filename )
{
	//load the file
	TiXmlDocument doc( filename );
	doc.LoadFile();

	//check for error
	if( doc.Error() )
		return false;

	//get the root element
	TiXmlNode*	root = doc.RootElement();

	//parse the entity
	ParseEntities( root->FirstChildElement( "Entity" ) );

	//parse the light
	ParseLights( root->FirstChildElement( "Light" ) );

	//parse kd-tree
	ParseKDTree( root->FirstChildElement( "KDTree" ) );

	//clear the doc
	doc.Clear();

	//load the vertex buffer
	LoadVertexBuffer();
	//load attribute buffer
	LoadAttributeBuffer();

	//copy memory to gpu
	CopyMemoryToGPU();

	return true;
}

//Parse Entity
void CustomScene::ParseEntities( TiXmlElement* node )
{
	//check if there is entity
	if( node == NULL )
		return;

	//get the number of entities
	m_iEntityNumber = 0;
	TiXmlElement* current = node;
	do
	{
		current = current->NextSiblingElement( "Entity" );
		m_iEntityNumber++;
	}while( current != NULL );

	//allocate the space for the entities
	SAFE_DELETEARRAY(m_pMeshList);
	m_pMeshList = new CustomMesh[m_iEntityNumber];

	//Parse the meshes
	current = node;
	int currentIndex = 0;
	do
	{
		//parse the current entity
		ParseEntity( current , &m_pMeshList[currentIndex] );
		
		//get to the next sibling
		current = current->NextSiblingElement( "Entity" );
		currentIndex++;
	}while( current != NULL );
}

//parse one single entity
void CustomScene::ParseEntity( TiXmlElement* node , CustomMesh* mesh )
{
	//get the filename
	const char*	filename = node->Attribute( "FileName" );

	//Load the mesh
	mesh->LoadObjFromFile( filename );

	//Get the matrix
	const char* matBuf = node->FirstChildElement( "WorldMatrix" )->Attribute( "matrix" );
	mesh->SetWorldMatrix( ParseMatrix( matBuf ) );
}

//Parse the lights
void CustomScene::ParseLights( TiXmlElement* node )
{
	//set default direction
	m_LightPos[0] = D3DXVECTOR4( 1 , 0 , 0 , 1 );
	m_LightPos[1] = D3DXVECTOR4( -1 , 0 , 0 , 0 );

	m_LightNum = 1;
	
	if( node != 0 )
	{
		int index = 0;
		m_LightNum = 0;

		do
		{
			//load the data
			TiXmlElement* lightNode = node->FirstChildElement( "Position" );

			const char* vecBuf = lightNode->Attribute( "pos" );

			float pos[4];
			int i = 0 , k = 0 , j = 0;
			char buf[128];
			while( true )
			{
				if( vecBuf[i] != ' ' && vecBuf[i] != 0 )
					buf[k] = vecBuf[i];
				else
				{
					buf[k] = 0;
					pos[j] = (float)atof( buf );

					k = -1;
					j++;
				}

				if( vecBuf[i] == 0 )
					break;

				k++;
				i++;
			}

			m_LightPos[m_LightNum] = D3DXVECTOR4( pos[0] , pos[1] , pos[2] , pos[3] );

			m_LightNum++;

			node = node->NextSiblingElement( "Light" );

		}while( node );
	}
}

//Load kd-tree
void CustomScene::ParseKDTree( TiXmlElement* node )
{
	//load the data
	node = node->FirstChildElement( "kd-tree" );
	const char* filename = node->Attribute( "filename" );

	//load the kd-tree
	m_KDTree.LoadKDTree( filename );
}

//Draw scene
void CustomScene::DrawScene()
{
	if( m_pMeshList )
	{
		for( int i = 0 ; i < m_iEntityNumber ; i++ )
			m_pMeshList[i].DrawMesh();
	}
}

//Update the scene
void CustomScene::Update( D3DXMATRIX* composite , D3DXVECTOR3* eye )
{
	for( int i = 0 ; i < m_iEntityNumber ; i++ )
		m_pMeshList[i].Update( composite , eye );

	//update the direction light
	SetDirectionalLight();

	//update for kd-tree
	m_KDTree.Update( composite );
}

//Set directional light
void CustomScene::SetDirectionalLight()
{
	for( int i = 0 ; i < m_iEntityNumber ; i++ )
	{
		float light[16];
		memset( light , 0 , sizeof( float ) * 16 );
		for( int k = 0 ; k < m_LightNum ; k++ )
		{
			int offset = 4 * k;

			light[offset] = m_LightPos[k].x;
			light[offset+1] = m_LightPos[k].y;
			light[offset+2] = m_LightPos[k].z;
			light[offset+3] = m_LightPos[k].w;
		}

		m_pMeshList[i].SetLightPosition( light );
	}
}

//Parse matrix
D3DXMATRIX CustomScene::ParseMatrix( const char* matrixBuffer )
{
	//the length of the buffer
	int len = (int)strlen( matrixBuffer );

	//the matrix
	D3DXMATRIX mat;
	memset( &mat , 0 , sizeof( mat ) );
	float*	p = (float*)&mat;

	int index = 0;
	int offset = 0;
	int matIndex = 0;
	while( index <= len )
	{
		char buf[256];

		if( matrixBuffer[index] != ' ' && matrixBuffer[index] != 0 )
			buf[index-offset] = matrixBuffer[index];
		else
		{
			buf[index-offset] = 0;

			//update the matrix
			p[matIndex] = (float)atof( buf );

			//set the new offset
			offset = index + 1;
			matIndex++;
		}

		index++;
	}

	return mat;
}

//Get total vertex buffer
_float4* CustomScene::GetVertexBuffer()
{
	return m_pVertexBuffer;
}

//Get Vertex number
UINT CustomScene::GetVertexNumber()
{
	return m_iVertexNumber;
}

//get vertex stride
UINT CustomScene::GetVertexStride()
{
	return sizeof( Custom_Vertex );
}

//Load the vertex buffer
void CustomScene::LoadVertexBuffer()
{
	//get the number of total vertex
	m_iVertexNumber = 0;
	for( int i = 0 ; i < m_iEntityNumber ; i++ )
		m_iVertexNumber += m_pMeshList[i].GetVertexNumber();

	SAFE_DELETEARRAY(m_pVertexBuffer);
	SAFE_DELETEARRAY(m_pNormalBuffer);
	SAFE_DELETEARRAY(m_pTexCoordinateBuffer);

	//allocate the memory
	m_pVertexBuffer = new _float4[ m_iVertexNumber ];
	m_pNormalBuffer = new _float4[ m_iVertexNumber ];
	m_pTexCoordinateBuffer = new float[ m_iVertexNumber * 2 ];

	//copy the vertexes
	int offset = 0;
	for( int i = 0 ; i < m_iEntityNumber ; i++ )
	{
		//get the vertex buffer
		vector<Custom_Vertex> list = m_pMeshList[i].GetVertexBuffer();

		//the number of vertex
		int vertexnumber = m_pMeshList[i].GetVertexNumber();

		//get the world matrix
		D3DXMATRIX world = m_pMeshList[i].GetWorldMatrix();

		//copy the vertex
		for( int j = 0 ; j < vertexnumber; j++ )
		{
			m_pVertexBuffer[offset+j].x = world._11 * list[j].x + world._21 * list[j].y + world._31 * list[j].z + world._41;
			m_pVertexBuffer[offset+j].y = world._12 * list[j].x + world._22 * list[j].y + world._32 * list[j].z + world._42;
			m_pVertexBuffer[offset+j].z = world._13 * list[j].x + world._23 * list[j].y + world._33 * list[j].z + world._43;
			m_pVertexBuffer[offset+j].w = (float)(offset+j)/3+1;

			m_pNormalBuffer[offset+j].x = ( world._11 * list[j].n_x + world._21 * list[j].n_y + world._31 * list[j].n_z );
			m_pNormalBuffer[offset+j].y = ( world._12 * list[j].n_x + world._22 * list[j].n_y + world._32 * list[j].n_z );
			m_pNormalBuffer[offset+j].z = ( world._13 * list[j].n_x + world._23 * list[j].n_y + world._33 * list[j].n_z );
			m_pNormalBuffer[offset+j].w = 0;
			normalize( m_pNormalBuffer[offset+j] );

			m_pTexCoordinateBuffer[2*(offset+j)] = list[j].u;
			m_pTexCoordinateBuffer[2*(offset+j)+1] = list[j].v;
		}

		//update the offset
		offset += vertexnumber;
	}
}

//Load attribute buffer
void CustomScene::LoadAttributeBuffer()
{
	//the total number of materials
	m_iMaterialNumber = 0;
	m_iFaceNumber = 0;
	for( int i = 0 ; i < m_iEntityNumber ; i++ )
	{
		m_iMaterialNumber += m_pMeshList[i].GetMaterialNumber();
		m_iFaceNumber += m_pMeshList[i].GetVertexNumber();
	}
	
	//allocate the memory
	SAFE_DELETEARRAY( m_pMaterialBuffer );
	SAFE_DELETEARRAY( m_pAttributeBuffer );

	m_pMaterialBuffer = new Custom_Material*[m_iMaterialNumber];
	m_pAttributeBuffer = new int[m_iFaceNumber];
	int* offset = new int[m_iMaterialNumber];

	offset[0] = 0;
	int attriOffset = 0;
	for( int i = 0 ; i < m_iEntityNumber ; i++ )
	{
		//get the material buffer
		vector<Custom_Material*> mat = m_pMeshList[i].GetCustomMaterial();

		//get the size of current material
		int size = m_pMeshList[i].GetMaterialNumber();

		//scan the offset
		if( i != m_iEntityNumber - 1 )
			offset[i+1] = offset[i] + size;
	
		//copy the materials
		for( int k = 0 ; k < size ; k++ )
			m_pMaterialBuffer[offset[i]+k] = mat[k];

		//copy the attribute
		int				faceNum = m_pMeshList[i].GetVertexNumber() / 3;
		vector<int>		attributeBuffer = m_pMeshList[i].GetAttributeBuffer();
		for( int k = 0 ; k < faceNum ; k++ )
		{
			int add = attriOffset + k;
			m_pAttributeBuffer[add] = (attributeBuffer[k]!=-1)? (attributeBuffer[k] + offset[i]) : 0;
		}
		attriOffset += faceNum;
	}

	delete[] offset;
}

//Get kd-tree
KDTree*	CustomScene::GetKDTree()
{
	return &m_KDTree;
}

//Get Light dir
void CustomScene::GetLightPosition( float* dir , int index )
{
	dir[0] = m_LightPos[index].x;
	dir[1] = m_LightPos[index].y;
	dir[2] = m_LightPos[index].z;
	dir[3] = m_LightPos[index].w;
}

//get material buffer
Custom_Material** CustomScene::GetMaterialBuffer()
{
	return m_pMaterialBuffer;
}

//get attribute buffer
int* CustomScene::GetAttributeBuffer()
{
	return m_pAttributeBuffer;
}

//get normal buffer
_float4*	CustomScene::GetNormalBuffer()
{
	return m_pNormalBuffer;
}

//get texture coordinate buffer
float*	CustomScene::GetTextureCoodinateBuffer()
{
	return m_pTexCoordinateBuffer;
}

//get the number of the lights
int	 CustomScene::GetLightNumber()
{
	return ( m_LightPos[1].w == 0 )?1:2;
}

//copy memory to gpu
void CustomScene::CopyMemoryToGPU()
{
	//release the memory first
	SAFE_RELEASE_CUDA( m_cVertexBuffer );
	SAFE_RELEASE_CUDA( m_cNormalBuffer );
	SAFE_RELEASE_CUDA( m_cTexCoordinateBuffer );
	SAFE_RELEASE_CUDA( m_cAttributeBuffer );
	SAFE_RELEASE_CUDA( m_cMaterialBuffer );
	SAFE_RELEASE_CUDA( m_cLightBuffer );

	//copy buffers
	cudaMalloc( (void**)&m_cVertexBuffer , sizeof( float4 ) * m_iVertexNumber );
	cudaMalloc( (void**)&m_cNormalBuffer , sizeof( float4 ) * m_iVertexNumber );
	cudaMalloc( (void**)&m_cTexCoordinateBuffer , sizeof( float2 ) * m_iVertexNumber );
	cudaMalloc( (void**)&m_cAttributeBuffer , sizeof( int ) * m_iFaceNumber );
	cudaMalloc( (void**)&m_cMaterialBuffer , sizeof( float4 ) * 4 * m_iMaterialNumber );
	cudaMalloc( (void**)&m_cLightBuffer , sizeof( float4 ) * 2 );

	//copy the memory
	cudaMemcpy( (void*)m_cVertexBuffer , m_pVertexBuffer , sizeof( float4 ) * m_iVertexNumber , cudaMemcpyHostToDevice );
	cudaMemcpy( (void*)m_cNormalBuffer , m_pNormalBuffer , sizeof( float4 ) * m_iVertexNumber , cudaMemcpyHostToDevice );
	cudaMemcpy( (void*)m_cTexCoordinateBuffer , m_pTexCoordinateBuffer , sizeof( float2) * m_iVertexNumber , cudaMemcpyHostToDevice );
	cudaMemcpy( (void*)m_cAttributeBuffer , m_pAttributeBuffer , sizeof( int ) * m_iFaceNumber , cudaMemcpyHostToDevice );
	cudaMemcpy( (void*)m_cLightBuffer , m_LightPos , sizeof( float4 ) * 2 , cudaMemcpyHostToDevice );

	//copy material buffer
	int* texOffset = D3DResource::GetSingleton()->GetTextureOffset();
	float* mat = new float[ 16 * m_iMaterialNumber ];
	for( int i = 0 ; i < m_iMaterialNumber ; i++ )
	{
		for( int k = 0 ; k < 4 ; k++ )
			mat[16*i+k] = m_pMaterialBuffer[i]->m_Ambient[k];
		for( int k = 0 ; k < 4 ; k++ )
			mat[16*i+4+k] = m_pMaterialBuffer[i]->m_Diffuse[k];
		for( int k = 0 ; k < 4 ; k++ )
			mat[16*i+8+k] = m_pMaterialBuffer[i]->m_Specular[k];
		mat[16*i+11] = (float)m_pMaterialBuffer[i]->m_nPower;

		m_pMaterialBuffer[i]->m_iTextureIndex = D3DResource::GetSingleton()->GetTextureIndex( m_pMaterialBuffer[i]->m_DifTextureName.c_str() );
		mat[16*i+12] = (float)m_pMaterialBuffer[i]->m_iTextureIndex;
		mat[16*i+13] = m_pMaterialBuffer[i]->m_fReflect;
		mat[16*i+14] = m_pMaterialBuffer[i]->m_fRefract;
		mat[16*i+15] = m_pMaterialBuffer[i]->m_fRefractRate;
	}
	cudaMemcpy( (void*)m_cMaterialBuffer , mat , sizeof( float ) * 16 * m_iMaterialNumber , cudaMemcpyHostToDevice );
	delete[] mat;
}

//Get cuda vertex buffer
float4*	CustomScene::GetCUDAVertexBuffer()
{
	return m_cVertexBuffer;
}

//get normal buffer
float4*	CustomScene::GetCUDANormalBuffer()
{
	return m_cNormalBuffer;
}
//get texture coordinate buffer
float2*	CustomScene::GetCUDATexCoordinateBuffer()
{
	return m_cTexCoordinateBuffer;
}
//get attribute buffer
int* CustomScene::GetCUDAAttributeBuffer()
{
	return m_cAttributeBuffer;
}

//get light buffer
float4*	CustomScene::GetCUDALightBuffer()
{
	return m_cLightBuffer;
}

//get cuda material buffer
float4*	CustomScene::GetCUDAMaterialBuffer()
{
	return m_cMaterialBuffer;
}

//show triId in render target
void CustomScene::DrawTriangleID( LPDIRECT3DSURFACE9 rt , D3DXMATRIX* proj )
{
	//the device
	LPDIRECT3DDEVICE9 device = D3DResource::GetSingleton()->GetDevice();
	//the effect
	LPD3DXEFFECT effect = D3DResource::GetSingleton()->GetEffect( DEFAULT_SHADER_FOR_TRIID );

	//update the composite matrix
	effect->SetMatrix( "CompositeMatrix" , proj );

	//get the old render target
	LPDIRECT3DSURFACE9 oldRT;
	device->GetRenderTarget( 0 , &oldRT );

	//set the render target
	device->SetRenderTarget( 0 , rt );

	//set vertex format
	device->SetFVF( D3DFVF_XYZW );

	//clear the buffer
	device->Clear( 0 , NULL , D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER , D3DXCOLOR( 0 , 0 , 0 , 0 ) , 1.0f , 0 );

	//begin to draw
	device->BeginScene();

	effect->Begin(0,0);
	effect->BeginPass(0);

	device->DrawPrimitiveUP( D3DPT_TRIANGLELIST , m_iVertexNumber / 3  , (void*)m_pVertexBuffer , sizeof( float ) * 4 );

	effect->EndPass();
	effect->End();

	//end drawing
	device->EndScene();

	//present the buffer
	device->Present( 0 , 0 , 0 , 0 );
	
	//restore the old render target
	device->SetRenderTarget( 0 , oldRT );
	SAFE_RELEASE(oldRT);
}