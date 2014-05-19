/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	CustomMesh.cpp
 *
 *	Description:	Load mesh from obj file
 */

#include "stdafx.h"
#include "CustomMesh.h"
#include "define.h"
#include "D3DResource.h"
#include <fstream>

//constructor and destructor
CustomMesh::CustomMesh()
{
	InitializeDefault();
}

CustomMesh::~CustomMesh()
{
	ReleaseContent();
}

//Initialize default
void CustomMesh::InitializeDefault()
{
	//the default material
	Custom_Material* mat = new Custom_Material();
	mat->m_Ambient = _float4( 0 , 0 , 0 , 0 );
	mat->m_Diffuse = _float4( 0.8f , 0.8f , 0.8f , 0 );
	mat->m_nPower = 1;
	mat->m_Specular = _float4( 0 , 0 , 0 , 0 );
	m_Materials.push_back( mat );

	//set world matrix identity
	D3DXMatrixIdentity( &m_WorldMatrix );
}

//Parse material file
void CustomMesh::ParseMaterial( const char* filename )
{
	//open the file
	ifstream file( filename );

	if( file.is_open() == false )
		return;

	string str;
	Custom_Material*	currentMat = NULL;
	while( file>>str )
	{
		if( strcmp( str.c_str() , "newmtl" ) == 0 )
		{
			//push the material into the list
			if( currentMat != NULL )
				m_Materials.push_back( currentMat );

			//allocate the new material
			currentMat = new Custom_Material();

			//Load the name of the material
			file>>currentMat->m_MaterialName;
		}else if( currentMat != NULL )
		{
			if( strcmp( str.c_str() , "Ka" ) == 0 )
			{
				file>>currentMat->m_Ambient.r;
				file>>currentMat->m_Ambient.g;
				file>>currentMat->m_Ambient.b;
			}else if( strcmp( str.c_str() , "Kd" ) == 0 )
			{
				file>>currentMat->m_Diffuse.r;
				file>>currentMat->m_Diffuse.g;
				file>>currentMat->m_Diffuse.b;
			}else if( strcmp( str.c_str() , "Ks" ) == 0 )
			{
				file>>currentMat->m_Specular.r;
				file>>currentMat->m_Specular.g;
				file>>currentMat->m_Specular.b;
			}else if( strcmp( str.c_str() , "Ns" ) == 0 )
			{
				file>>currentMat->m_nPower;
			}else if( strcmp( str.c_str() , "map_Kd" ) == 0 )
			{
				char name[512];
				file>>name;

				char buf[512];
				strcpy_s( buf , 256 , m_MediaPath );
				strcat_s( buf , 256 , name );

				currentMat->m_DifTextureName = string(buf);

				//push the texture in
				D3DResource::GetSingleton()->PushTexture( buf );
			}else if( strcmp( str.c_str() , "reflect" ) == 0 )
			{
				file>>currentMat->m_fReflect;
			}else if( strcmp( str.c_str() , "refract" ) == 0 )
			{
				file>>currentMat->m_fRefract;
				file>>currentMat->m_fRefractRate;
			}
		}

		//ignore the rest of the line
		file.ignore( 1024 , '\n' );

		if( file.eof() )
			break;
	}

	//push the material into the list
	if( currentMat != NULL )
		m_Materials.push_back( currentMat );

	//close the file
	file.close();
}

//Release the material
void CustomMesh::ReleaseMaterial()
{
	vector<Custom_Material*>::iterator it = m_Materials.begin();

	while( it != m_Materials.end() )
	{
		delete *it;
		it++;
	}

	m_Materials.clear();
}

//Parse obj from file
void CustomMesh::LoadObjFromFile( const char* filename )
{
	//open file first
	ifstream file( filename );

	if( file.is_open() == false )
		return;

	//Parse the path of the filename
	int len = (int)strlen( filename );
	int i = len;
	while( i >= 0 )
	{
		i--;
		if( filename[i] == '/' || filename[i] == '\\' )
			break;
	}
	m_MediaPath[i+1] = 0;
	for( ; i >= 0 ; i-- )
		m_MediaPath[i] = filename[i];

	//parse the file
	string str;

	//current material index
	int	matIndex = -2;
	//current subset vertex number
	int curerentMatVertexCount = 0;

	//the vertex list
	vector<_float4>	positionBuffer;
	vector<_float4>	normalBuffer;
	vector<_float4>	texCoordBuffer;

	while( true )
	{
		//load the first string of current line
		file>>str;

		const char* pStr = str.c_str();

		if( pStr[0] == '#' )
		{
			//ignore the line
			file.ignore( 1024 , '\n' );
		}else if( strcmp( pStr , "mtllib" ) == 0 )
		{
			//parse the material
			file>>str;
			
			char buf[512];
			strcpy_s( buf , 256 , m_MediaPath );
			strcat_s( buf , 256 , str.c_str() );

			ParseMaterial( buf );
		}else if( strcmp( pStr , "v" ) == 0 )
		{
			//parse the position
			_float4 pos;
			file>>pos.x;
			file>>pos.y;
			file>>pos.z;

			//push the position into the buffer
			positionBuffer.push_back( pos );
		}else if( strcmp( pStr , "vt" ) == 0 )
		{
			//parse the texture coordinate
			_float4 texCoord;
			file>>texCoord.x;
			file>>texCoord.y;

			//push the texture coordinate into the buffer
			texCoordBuffer.push_back( texCoord );
		}else if( strcmp( pStr , "vn" ) == 0 )
		{
			//parse the normal
			_float4 normal;
			file>>normal.x;
			file>>normal.y;
			file>>normal.z;

			//push the normal into the buffer
			normalBuffer.push_back( normal );
		}else if( strcmp( pStr , "usemtl" ) == 0 )
		{
			//get the material name
			string matName;
			file>>matName;

			if( matIndex >= -1 )
			{
				m_SubsetVertexCount.push_back( curerentMatVertexCount );
				curerentMatVertexCount = 0 ;
			}
			
			//get the material index
			matIndex = GetMaterialIndex( matName.c_str() );
		}else if( strcmp( pStr , "f" ) == 0 )
		{
			for( int i = 0 ; i < 3 ; i++ )
			{
				file>>str;

				//the vertex
				Custom_Vertex v;

				//parse the index
				ParseIndex( &v , str.c_str() , positionBuffer , normalBuffer , texCoordBuffer );

				//push into the buffer
				m_VertexBuffer.push_back( v );
			}

			//update the material buffer
			m_AttributeBuffer.push_back( matIndex );

			//increase vertex count
			curerentMatVertexCount+=3;
		}else
		{
			//ignore the current line
			file.ignore( 1024 , '\n' );
		}

		//break out if there is no content
		if( file.eof() )
			break;
	}

	m_SubsetVertexCount.push_back( curerentMatVertexCount );

	//clear the buffers
	positionBuffer.clear();
	normalBuffer.clear();
	texCoordBuffer.clear();

	//create vertex buffer for d3d
	CreateVertexBuffer();

	//close the file
	file.close();
}

//create the vertex buffer
void CustomMesh::CreateVertexBuffer()
{
	//get the device first
	LPDIRECT3DDEVICE9	device = D3DResource::GetSingleton()->GetDevice();

	//create the vertex buffer
	int vertexNum = (int)m_VertexBuffer.size();
	device->CreateVertexBuffer( sizeof( Custom_Vertex ) * vertexNum , 0 , CustomVertexFormatFNT , D3DPOOL_MANAGED , &m_lpVertexes , NULL );

	//lock the buffer
	void*	pData = 0;
	m_lpVertexes->Lock( 0 , sizeof( Custom_Vertex ) * vertexNum , &pData , D3DLOCK_DISCARD );

	//fill the buffer
	void*	srcData = (void*)(&(*m_VertexBuffer.begin()));
	memcpy( pData , srcData , sizeof( Custom_Vertex ) * vertexNum );

	//unlock the buffer
	m_lpVertexes->Unlock();
}

//Get material index
int	CustomMesh::GetMaterialIndex( const char* matName )
{
	//get the iterator
	vector<Custom_Material*>::iterator it = m_Materials.begin();
	
	int index = 0;
	while( it != m_Materials.end() )
	{
		if( strcmp( (*it)->m_MaterialName.c_str() , matName ) == 0 )
			return index;

		index++;
		it++;
	}

	return -1;
}

//Parse the index
void	CustomMesh::ParseIndex( Custom_Vertex* vertex , const char* buffer , vector<_float4>& pos , vector<_float4>& nor , vector<_float4>& tex )
{
	//the index for the vertex
	int posIndex = -1 , norIndex = -1 , tcIndex = -1;

	//the buffer for the interger
	char intBuf[128];
	
	//load the first data
	int index = 0;
	int curIndex = 0;
	while( buffer[index] != '/' && buffer[index] != 0 )
	{
		//copy the buffer
		intBuf[curIndex] = buffer[index];

		//increase index
		curIndex++;
		index++;
	}

	//Load the position
	intBuf[curIndex] = 0;
	posIndex = atoi( intBuf ) - 1;
	
	//Load the next buffer if nessassary
	if( buffer[index] == '/' )
	{
		//increase the index
		index++;

		//load the next data
		if( buffer[index] != '/' )
		{
			curIndex = 0;
			while( buffer[index] != '/' && buffer[index] != 0 )
			{
				//copy the buffer
				intBuf[curIndex] = buffer[index];

				//increase index
				curIndex++;
				index++;
			}

			//close the integer
			intBuf[curIndex] = 0;

			tcIndex = atoi( intBuf ) - 1;
		}

		//Load the last data
		if( buffer[index] == '/' )
		{
			index++;
			curIndex = 0;
			while(	buffer[index] != '/' && buffer[index] != 0 )
			{
				//copy the buffer
				intBuf[curIndex] = buffer[index];

				//increase index
				curIndex++;
				index++;
			}

			//close the integer
			intBuf[curIndex] = 0;

			norIndex = atoi( intBuf ) - 1;
		}
	}

	//set the data
	vertex->x = pos[posIndex].x; vertex->y = pos[posIndex].y; vertex->z = pos[posIndex].z;
	if( norIndex >= 0 )
	{
		vertex->n_x = nor[norIndex].x;
		vertex->n_y = nor[norIndex].y;
		vertex->n_z = nor[norIndex].z;
	}
	if( tcIndex >= 0 )
	{
		vertex->u = tex[tcIndex].x;
		vertex->v = tex[tcIndex].y;
	}
}

//Render the mesh
void CustomMesh::DrawMesh()
{
	//the data pointer
	Custom_Vertex*	data = (Custom_Vertex*) &(*m_VertexBuffer.begin());

	//the subset number
	int subsetNum = (int)m_SubsetVertexCount.size();

	//current offset
	int offset = 0;

	//get the effect and the device
	LPD3DXEFFECT		effect = D3DResource::GetSingleton()->GetEffect( DEFAULT_SHADER_FOR_ENTITY );
	LPDIRECT3DDEVICE9	device = D3DResource::GetSingleton()->GetDevice();

	//udpate matrix
	effect->SetMatrix( "CompositeMatrix" , &m_CompositeMatrix );
	HRESULT hr = effect->SetMatrix( "WorldMatrix" , &m_WorldMatrix );
	//update eye position
	effect->SetVector( "EyePosition" , &m_EyePosition );

	for( int i = 0 ; i < subsetNum ; i++ )
	{
		//Get the material index
		int matIndex = m_AttributeBuffer[offset/3];

		//Update material
		UpdateMaterial( matIndex );
		
		//current vertex count
		int count = m_SubsetVertexCount[i] / 3;

		//begin the pass
		effect->Begin( 0 , 0 );
		effect->BeginPass(0);

		//draw the mesh
		device->SetStreamSource( 0 , m_lpVertexes , 0 , sizeof( Custom_Vertex ) );
		device->DrawPrimitive( D3DPT_TRIANGLELIST , offset , count );

		//end
		effect->EndPass();
		effect->End();

		//update offset
		offset += m_SubsetVertexCount[i];
	}
}

//Release content
void CustomMesh::ReleaseContent()
{
	m_VertexBuffer.clear();
	m_AttributeBuffer.clear();
	m_SubsetVertexCount.clear();

	//release vertex buffer
	SAFE_RELEASE( m_lpVertexes );

	//release material
	ReleaseMaterial();
}

//Update matrix
void CustomMesh::Update( D3DXMATRIX* composite , D3DXVECTOR3* eye )
{
	//update the matrix and eye position
	m_CompositeMatrix = m_WorldMatrix * (*composite);
	m_EyePosition = D3DXVECTOR4(*eye);
}

//Update material for shader effect
void CustomMesh::UpdateMaterial( int matIndex )
{
	//Get the material
	matIndex = max( 0 , matIndex );
	Custom_Material* mat = m_Materials[matIndex];

	//get the effect
	LPD3DXEFFECT effect = D3DResource::GetSingleton()->GetEffect( DEFAULT_SHADER_FOR_ENTITY );

	//update diffuse color
	effect->SetFloatArray( "DiffuseColor" , (float*)&mat->m_Diffuse , 4 );

	//update ambient color
	effect->SetFloatArray( "AmbientColor" , (float*)&mat->m_Ambient , 4 );

	//update specular color
	effect->SetFloatArray( "SpecularColor" , (float*)&mat->m_Specular , 4 );
	effect->SetInt( "SpecularPower" , mat->m_nPower );

	//update the texture
	mat->m_iTextureIndex = D3DResource::GetSingleton()->GetTextureIndex( mat->m_DifTextureName.c_str() );
	mat->m_Texture = D3DResource::GetSingleton()->GetTexture( mat->m_DifTextureName.c_str() );
	effect->SetTexture( "DiffuseTex" , mat->m_Texture );

	//get the device
	LPDIRECT3DDEVICE9	device = D3DResource::GetSingleton()->GetDevice();

	//set the technique
	if( mat->m_Texture )
	{
		effect->SetTechnique( "PNT_Tec" );
		device->SetFVF( CustomVertexFormatFNT );
	}
	else
	{
		effect->SetTechnique( "PN_Tec" );
		device->SetFVF( CustomVertexFormatFN );
	}
}

//Set directional light
void CustomMesh::SetLightPosition( float* pos )
{
	//get the effect
	LPD3DXEFFECT effect = D3DResource::GetSingleton()->GetEffect( DEFAULT_SHADER_FOR_ENTITY );

	effect->SetMatrix( "LightPosition" , (D3DXMATRIX*)pos );
}

//Set world matrix
void CustomMesh::SetWorldMatrix( D3DXMATRIX& mat )
{
	m_WorldMatrix = mat;
}

//Get the vertex buffer
vector<Custom_Vertex>& CustomMesh::GetVertexBuffer()
{
	return m_VertexBuffer;
}

//Get the number of vertex
UINT CustomMesh::GetVertexNumber()
{
	return (UINT)m_VertexBuffer.size();
}

//Get world matrix
D3DXMATRIX&	CustomMesh::GetWorldMatrix()
{
	return m_WorldMatrix;
}

//Get atrribute buffer
vector<int>& CustomMesh::GetAttributeBuffer()
{
	return m_AttributeBuffer;
}

//Get Material list
vector<Custom_Material*>& CustomMesh::GetCustomMaterial()
{
	return m_Materials;
}

//Get Material number
UINT CustomMesh::GetMaterialNumber()
{
	return (UINT)m_Materials.size();
}