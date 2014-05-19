/*
 *	FileName:	KDTree.cpp
 *
 *	Programmer:	Jiayin Cao
 */

#include <fstream>
#include "KDTree.h"
#include "define.h"
#include "D3DResource.h"

using namespace std;

//constructor and destructor
KDTree::KDTree()
{
	//set default value
	InitializeDefault();
}
KDTree::~KDTree()
{
}

//initiailze default
void KDTree::InitializeDefault()
{
	m_KDTreeBuffer = NULL;
	m_TriIndexBuffer = NULL;
	m_lpVertexes = NULL;
	m_iNodeNumber = 0;
	m_iTriNumber = 0;
	m_cKDTreeBuffer = 0;
	m_cTriIndexBuffer = 0;
	m_cOffsetBuffer = 0;
	m_pOffsetBuffer = 0;
	m_iBBOffset = 0;
}

//release the content
void KDTree::ReleaseContent()
{
	SAFE_RELEASE_CUDA( m_cKDTreeBuffer );
	SAFE_RELEASE_CUDA( m_cTriIndexBuffer );
	SAFE_RELEASE_CUDA( m_cOffsetBuffer );
	SAFE_DELETEARRAY( m_KDTreeBuffer );
	SAFE_DELETEARRAY( m_TriIndexBuffer );
	SAFE_DELETEARRAY( m_pOffsetBuffer );
	SAFE_RELEASE( m_lpVertexes );
	m_iNodeNumber = 0;
	m_iTriNumber = 0;
}

//load the kd-tree from file
void KDTree::LoadKDTree( const char* filename )
{
	//open the file
	ifstream file( filename , ios::binary );

	//if the file is not opened, return
	if( file.is_open() == false )
		return;

	//Load the node number
	file>>m_iNodeNumber;

	//allocate the memory
	SAFE_DELETEARRAY( m_KDTreeBuffer );

	int size = m_iNodeNumber * 16;
	m_KDTreeBuffer = new float[ size ];

	//load the memory
	for( int i = 0 ; i < size; i++ )
		file>>m_KDTreeBuffer[i];

	//load the primitives number
	file>>m_iTriNumber;

	//malloc the data
	m_pOffsetBuffer = new int[ m_iNodeNumber ];

	if( m_iTriNumber < 0 )
	{
		for( int i = 0 ; i < m_iNodeNumber ; i++ )
			file>>m_pOffsetBuffer[i];

		//load the triangle number again
		file>>m_iTriNumber;
	}else
	{
		for( int i = 0 ; i < m_iNodeNumber ; i++ )
			m_pOffsetBuffer[i] = (int)m_KDTreeBuffer[16*i+7];
	}

	//allocate the memory of triangle index
	m_TriIndexBuffer = new int[m_iTriNumber];

	for( int i = 0 ; i < m_iTriNumber ; i++ )
		file>>m_TriIndexBuffer[i];

	//close the file
	file.close();

	//copy the memory to gpu
	CopyMemoryToGPU();

	//create vertex buffer
	CreateVertexBuffer();
}

//copy memory to gpu
void KDTree::CopyMemoryToGPU()
{
	SAFE_RELEASE_CUDA( m_cKDTreeBuffer );
	SAFE_RELEASE_CUDA( m_cTriIndexBuffer );
	SAFE_RELEASE_CUDA( m_cOffsetBuffer );

	//create cuda memory
	cudaMalloc( (void**)&m_cKDTreeBuffer , m_iNodeNumber * sizeof( float ) * 16 );
	cudaMalloc( (void**)&m_cOffsetBuffer , m_iNodeNumber * sizeof( int ) );
	cudaMalloc( (void**)&m_cTriIndexBuffer , m_iTriNumber * sizeof( int ) );

	//copy the data to gpu
	cudaMemcpy( (void*)m_cKDTreeBuffer , m_KDTreeBuffer , m_iNodeNumber * sizeof( float ) * 16 , cudaMemcpyHostToDevice );
	cudaMemcpy( (void*)m_cTriIndexBuffer , m_TriIndexBuffer , m_iTriNumber * sizeof( int ) , cudaMemcpyHostToDevice );
	cudaMemcpy( (void*)m_cOffsetBuffer , m_pOffsetBuffer , m_iNodeNumber * sizeof( int ) , cudaMemcpyHostToDevice );
}

//save kd-tree for debugging
void KDTree::SaveKDTree( const char* filename )
{
	//open the file
	ofstream file( filename );

	//output the number of node
	file<<m_iNodeNumber<<endl;

	//output the nodes
	int offset = 0;
	for( int i = 0 ; i < m_iNodeNumber ; i++ )
	{
		file<<"index    \t"<<i<<endl;
		file<<"parent   \t"<<m_KDTreeBuffer[offset]<<endl;
		file<<"left     \t"<<m_KDTreeBuffer[offset+1]<<endl;
		file<<"right    \t"<<m_KDTreeBuffer[offset+2]<<endl;
		file<<"triNum   \t"<<m_KDTreeBuffer[offset+3]<<endl;
		file<<"splitAxis\t"<<m_KDTreeBuffer[offset+4]<<endl;
		file<<"splitPos \t"<<m_KDTreeBuffer[offset+5]<<endl;
		file<<"depth    \t"<<m_KDTreeBuffer[offset+6]<<endl;
		file<<"offset   \t"<<m_KDTreeBuffer[offset+7]<<endl;
		file<<endl;

		offset += 16;
	}

	//close the file
	file.close();
}

//draw kd-tree
void KDTree::DrawKDTree( int level )
{
	//get the device of d3d
	LPDIRECT3DDEVICE9 device = D3DResource::GetSingleton()->GetDevice();

	//Get the current effect
	LPD3DXEFFECT effect = D3DResource::GetSingleton()->GetEffect( DEFAULT_SHADER_FOR_KDTREE );

	//set the fvf for the device
	device->SetFVF( KDTree_Vertex::KD_FVF );

	effect->Begin( 0 , 0 );
	effect->BeginPass(0);

	//draw the mesh
	device->SetStreamSource( 0 , m_lpVertexes , 0 , sizeof( KDTree_Vertex ) );
	device->DrawPrimitive( D3DPT_LINELIST , 0 , m_iBBOffset / 2 );

	effect->EndPass();
	effect->End();
}

//push current bounding box into the buffer
void KDTree::PushBoundingBox( int index , int level , float* p )
{
	//get the kd-tree node pointer
	float*	node = m_KDTreeBuffer + index * 16;

	//check if the node is in the current level
	if( level != -1 )
	{
		if( node[6] == level )
		{
			//push a bounding box into the buffer
			PushBoundingBox( ((KDTree_Vertex*)p) + m_iBBOffset , node + 8 );
		}else
		{
			if( node[4] >= 0 )
			{
				PushBoundingBox( (int)node[1] , level , p );
				PushBoundingBox( (int)node[2] , level , p );
			}else
				PushBoundingBox( ((KDTree_Vertex*)p) + m_iBBOffset , node + 8 );
		}
	}else
	{
		if( node[4] >= 0 )
		{
			PushBoundingBox( (int)node[1] , level , p );
			PushBoundingBox( (int)node[2] , level , p );
		}else
			PushBoundingBox( ((KDTree_Vertex*)p) + m_iBBOffset , node + 8 );
	}
}

//push bounding box
void KDTree::PushBoundingBox( KDTree_Vertex* dest , float* boundingBox )
{
	//the max and min
	float* min = boundingBox;
	float* max = boundingBox + 4;

	//update the dest
	dest[0].x = min[0];	dest[0].y = min[1];	dest[0].z = min[2];
	dest[1].x = max[0];	dest[1].y = min[1];	dest[1].z = min[2];
	dest[2].x = max[0];	dest[2].y = min[1];	dest[2].z = min[2];
	dest[3].x = max[0];	dest[3].y = min[1];	dest[3].z = max[2];
	dest[4].x = max[0];	dest[4].y = min[1];	dest[4].z = max[2];
	dest[5].x = min[0];	dest[5].y = min[1];	dest[5].z = max[2];
	dest[6].x = min[0];	dest[6].y = min[1];	dest[6].z = max[2];
	dest[7].x = min[0];	dest[7].y = min[1];	dest[7].z = min[2];
	dest[8].x = min[0];	dest[8].y = max[1];	dest[8].z = min[2];
	dest[9].x = max[0];	dest[9].y = max[1];	dest[9].z = min[2];
	dest[10].x = max[0];	dest[10].y = max[1];	dest[10].z = min[2];
	dest[11].x = max[0];	dest[11].y = max[1];	dest[11].z = max[2];
	dest[12].x = max[0];	dest[12].y = max[1];	dest[12].z = max[2];
	dest[13].x = min[0];	dest[13].y = max[1];	dest[13].z = max[2];
	dest[14].x = min[0];	dest[14].y = max[1];	dest[14].z = max[2];
	dest[15].x = min[0];	dest[15].y = max[1];	dest[15].z = min[2];
	dest[16].x = min[0];	dest[16].y = min[1];	dest[16].z = min[2];
	dest[17].x = min[0];	dest[17].y = max[1];	dest[17].z = min[2];
	dest[18].x = min[0];	dest[18].y = min[1];	dest[18].z = max[2];
	dest[19].x = min[0];	dest[19].y = max[1];	dest[19].z = max[2];
	dest[20].x = max[0];	dest[20].y = min[1];	dest[20].z = max[2];
	dest[21].x = max[0];	dest[21].y = max[1];	dest[21].z = max[2];
	dest[22].x = max[0];	dest[22].y = min[1];	dest[22].z = min[2];
	dest[23].x = max[0];	dest[23].y = max[1];	dest[23].z = min[2];

	m_iBBOffset += 24;
}

//Update the scene
void KDTree::Update( D3DXMATRIX* composite )
{
	//Get the current effect
	LPD3DXEFFECT effect = D3DResource::GetSingleton()->GetEffect( DEFAULT_SHADER_FOR_KDTREE );

	//set the matrix
	effect->SetMatrix( "CompositeMatrix" , composite );
}

//the buffer of the kd-tree
float* KDTree::GetBuffer()
{
	return m_KDTreeBuffer;
}

//get the index buffer
int* KDTree::GetIndexBuffer()
{
	return m_TriIndexBuffer;
}

//get cuda vertex buffer
float4*	KDTree::GetCUDABuffer()
{
	return m_cKDTreeBuffer;
}

//get the cuda index buffer
int* KDTree::GetCUDAIndexBuffer()
{
	return m_cTriIndexBuffer;
}

//get offset buffer
int* KDTree::GetOffsetBuffer()
{
	return m_pOffsetBuffer;
}

//get cuda offset buffer
int* KDTree::GetCUDAOffsetBuffer()
{
	return m_cOffsetBuffer;
}

//create vertex buffer
void KDTree::CreateVertexBuffer()
{
	//allocate the data
	const int size = 1024 * 128 * 64;
	KDTree_Vertex* data = new KDTree_Vertex[size];

	//push the bounding box first
	PushBoundingBox( 0 , -1 , (float*)data );

	//get the device first
	LPDIRECT3DDEVICE9	device = D3DResource::GetSingleton()->GetDevice();

	//create the vertex buffer
	device->CreateVertexBuffer( sizeof( KDTree_Vertex ) * m_iBBOffset , 0 , KDTree_Vertex::KD_FVF , D3DPOOL_MANAGED , &m_lpVertexes , NULL );

	//lock the buffer
	void*	pData = 0;
	m_lpVertexes->Lock( 0 , sizeof( KDTree_Vertex ) * m_iBBOffset , &pData , D3DLOCK_DISCARD );

	//fill the buffer
	memcpy( pData , (void*)data , sizeof( KDTree_Vertex ) * m_iBBOffset );

	//unlock the buffer
	m_lpVertexes->Unlock();

	//release the buffer
	SAFE_DELETEARRAY( data );
}