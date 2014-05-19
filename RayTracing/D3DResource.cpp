/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	D3DResource.cpp
 */

#include "D3DResource.h"
#include "define.h"

//the sum for scan
extern int*	g_ScanSum[2];

//set the singleton pointer null
D3DResource* D3DResource::m_pSingleton = 0;

//constructor and destructor
D3DResource::D3DResource()
{
	m_pTextureOffset = 0;
	m_pCustomTexture = 0;
	m_cCustomTexture = 0;
	m_cTextureOffset = 0;

	cudaMalloc( (void**)&g_ScanSum[0] , sizeof( int ) * 262144 );
	cudaMalloc( (void**)&g_ScanSum[1] , sizeof( int ) * 512 );
}
D3DResource::~D3DResource()
{
	SAFE_RELEASE_CUDA( g_ScanSum[0] );
	SAFE_RELEASE_CUDA( g_ScanSum[1] );
}

//get the singleton
D3DResource* D3DResource::GetSingleton()
{
	if( m_pSingleton == 0 )
		m_pSingleton = new D3DResource();
	
	return m_pSingleton;
}

//destroy singleton
void D3DResource::Destroy()
{
	if( m_pSingleton != 0 )
	{
		m_pSingleton->ReleaseContent(true);	
		delete m_pSingleton;
	}
}

//Load the default shader effect
void D3DResource::LoadDefaultShader()
{
	if( m_lpd3dDevice == NULL )
		return;

	//load default shader
	LPD3DXEFFECT	effect = NULL;
	D3DXCreateEffectFromFile( m_lpd3dDevice , DEFAULT_SHADER_FOR_ENTITY_FILENAME , NULL , NULL , D3DXFX_NOT_CLONEABLE , NULL , &effect , NULL );

	if( effect )
		m_Effects.push_back( effect );
	else
	{
		WCHAR buf[1024];
		wsprintf( buf , L"Miss shader file %ws." , DEFAULT_SHADER_FOR_ENTITY_FILENAME );
		MessageBox( NULL , buf , L"Error" , 0 );
		PostQuitMessage(0);
	}

	effect = NULL;
	D3DXCreateEffectFromFile( m_lpd3dDevice , DEFAULT_SHADER_FOR_KDTREE_FILENAME , NULL , NULL , D3DXFX_NOT_CLONEABLE , NULL , &effect , NULL );
	if( effect )
		m_Effects.push_back( effect );
	else
	{
		WCHAR buf[1024];
		wsprintf( buf , L"Miss shader file %ws." , DEFAULT_SHADER_FOR_KDTREE_FILENAME );
		MessageBox( NULL , buf , L"Error" , 0 );
		PostQuitMessage(0);
	}

	effect = NULL;
	D3DXCreateEffectFromFile( m_lpd3dDevice , DEFAULT_SHADER_FOR_TRIID_FILENAME , NULL , NULL , D3DXFX_NOT_CLONEABLE , NULL , &effect , NULL );
	if( effect )
		m_Effects.push_back( effect );
	else
	{
		WCHAR buf[1024];
		wsprintf( buf , L"Miss shader file %ws." , DEFAULT_SHADER_FOR_TRIID_FILENAME );
		MessageBox( NULL , buf , L"Error" , 0 );
		PostQuitMessage(0);
	}
}

//Get shader effect
LPD3DXEFFECT D3DResource::GetEffect( int index )
{
	if( index < 0 || index >= (int) m_Effects.size() )
		return NULL;

	return m_Effects[index];
}

//Load the textures
void D3DResource::LoadTexture()
{
	if( m_TextureList.empty() || m_Textures.empty() == false )
		return;

	//allocate the memory
	SAFE_DELETEARRAY(m_pTextureOffset);
	SAFE_DELETEARRAY(m_pCustomTexture);
	m_pTextureOffset = new int[m_TextureList.size()];

	//first check if the file is already exist
	vector<string>::iterator it = m_TextureList.begin();

	int totalMemSize = 0;
	int index = 0;
	vector<_float4*> textureMem;

	while( it != m_TextureList.end() )
	{
		//load the texture
		LPDIRECT3DTEXTURE9	texture;
		D3DXCreateTextureFromFileA( m_lpd3dDevice , it->c_str() , &texture );

		//create custom memory
		_float4* mem = CopyTextureMemory( texture );

		//push it into the texture memory
		textureMem.push_back( mem );

		//set the offset
		m_pTextureOffset[index] = totalMemSize;
		totalMemSize += (int)mem[0].x;

		//push the texture in
		m_Textures.push_back( texture );
		
		//update the iterator
		it++;
		index++;
	}

	//allocate the memory
	m_pCustomTexture = new _float4[totalMemSize];

	//copy the memory
	vector<_float4*>::iterator it2 = textureMem.begin();
	index = 0;

	while( it2 != textureMem.end() )
	{
		//the data
		_float4* data = *it2;

		//copy the memory
		int size = (int)data[0].x;

		//copy the memory
		int offset = m_pTextureOffset[index];

		//copy the memory
		memcpy( m_pCustomTexture + offset , data , size * sizeof( _float4 ) );

		//delete the previous data
		delete[] data;

		//udpate the texture memory
		it2++;
		index++;
	}

	//copy to cuda memory
	cudaMalloc( (void**)&m_cCustomTexture , sizeof( float4 ) * totalMemSize );
	cudaMemcpy( m_cCustomTexture , m_pCustomTexture , sizeof( float4 ) * totalMemSize , cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&m_cTextureOffset , sizeof( int ) * m_TextureList.size() );
	cudaMemcpy( m_cTextureOffset , m_pTextureOffset , sizeof( int ) * m_TextureList.size() , cudaMemcpyHostToDevice );
}

//Load content
void D3DResource::LoadContent()
{
	//load textures
	LoadTexture();

	//load default shaders
	LoadDefaultShader();
}

//push filename into the texture list
void D3DResource::PushTexture( const char* filename )
{
	//first check if the file is already exist
	vector<string>::iterator it = m_TextureList.begin();

	while( it != m_TextureList.end() )
	{
		if( strcmp( (*it).c_str() , filename ) == 0 )
			return;
		it++;
	}

	m_TextureList.push_back( string( filename ) );
}

//get texture
LPDIRECT3DTEXTURE9 D3DResource::GetTexture( const char* filename )
{
	//first check if the file is already exist
	vector<string>::iterator it = m_TextureList.begin();

	int index = 0;
	while( it != m_TextureList.end() )
	{
		if( strcmp( (*it).c_str() , filename ) == 0 )
			return m_Textures[index];

		it++;
		index++;
	}

	return NULL;
}

//get the index of the texture
UINT D3DResource::GetTextureIndex( const char* filename )
{
	//first check if the file is already exist
	vector<string>::iterator it = m_TextureList.begin();

	UINT index = 0;
	while( it != m_TextureList.end() )
	{
		if( strcmp( (*it).c_str() , filename ) == 0 )
			return index;

		it++;
		index++;
	}

	return -1;
}

//set device
void D3DResource::SetDevice( LPDIRECT3DDEVICE9 device )
{
	m_lpd3dDevice = device;
}

//Unload content
void D3DResource::ReleaseContent( bool releaseTexture )
{
	//release default shader
	ReleaseDefaultShader();

	//release the textures
	if( releaseTexture )
		ReleaseTexture();
}

//Get the device of d3d
LPDIRECT3DDEVICE9	D3DResource::GetDevice()
{
	return m_lpd3dDevice;
}

//Release default shader
void D3DResource::ReleaseDefaultShader()
{
	vector<LPD3DXEFFECT>::iterator it = m_Effects.begin();

	while( it != m_Effects.end() )
	{
		SAFE_RELEASE( *it );
		it++;
	}

	m_Effects.clear();
}

//Release texture
void D3DResource::ReleaseTexture()
{
	vector<LPDIRECT3DTEXTURE9>::iterator it = m_Textures.begin();

	while( it != m_Textures.end() )
	{
		SAFE_RELEASE( *it );
		it++;
	}

	m_Textures.clear();

	//delete the texture memory
	SAFE_DELETEARRAY(m_pCustomTexture);
	SAFE_DELETEARRAY(m_pTextureOffset);

	SAFE_RELEASE_CUDA( m_cCustomTexture );
	SAFE_RELEASE_CUDA( m_cTextureOffset );
}

//copy the memory to a custom memory
_float4*	D3DResource::CopyTextureMemory(LPDIRECT3DTEXTURE9 texture)
{
	//get the description first
	D3DSURFACE_DESC desc;
	texture->GetLevelDesc( 0 , &desc );

	//allocate the memory
	int size = desc.Width * desc.Height + 1;
	_float4* mem = new _float4[ size ];

	//set the header
	mem[0].x = (float)size;
	mem[0].y = (float)desc.Width;
	mem[0].z = (float)desc.Height;

	//lock the surface
	D3DLOCKED_RECT d3d_Rect;
	texture->LockRect( 0 , &d3d_Rect , NULL , D3DLOCK_READONLY );
	DWORD* pData = (DWORD*)d3d_Rect.pBits;

	//copy the data
	for( int i = 1 ; i < size ; i++ )
	{
		//copy the data
		int x = ( i - 1 ) % desc.Width;
		int y = ( i - 1 ) / desc.Width;

		//copy the pixel
		mem[i].r = (float)(((pData[ y * d3d_Rect.Pitch / sizeof( DWORD ) + x ])>>16)&0x000000ff) / 255.0f;
		mem[i].g = (float)(((pData[ y * d3d_Rect.Pitch / sizeof( DWORD ) + x ])>>8)&0x000000ff) / 255.0f;
		mem[i].b = (float)(((pData[ y * d3d_Rect.Pitch / sizeof( DWORD ) + x ]))&0x000000ff) / 255.0f;
	}

	//unlock the rect
	texture->UnlockRect(0);

	return mem;
}

//get custom texture
_float4*	D3DResource::GetCustomTexture()
{
	return m_pCustomTexture;
}

//get texture offset
int*	D3DResource::GetTextureOffset()
{
	return m_pTextureOffset;
}

//get cuda texture
float4*	 D3DResource::GetCUDATexture()
{
	return m_cCustomTexture;
}

//get cuda texture offset
int* D3DResource::GetCUDATextureOffset()
{
	return m_cTextureOffset;
}

//whether the resource is ready
bool D3DResource::ResourceReady()
{
	return m_Effects.size() == 3;
}