/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	D3DResource.h
 */

#pragma once

#include <d3d9.h>
#include <d3dx9.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "datatype.h"
using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////////
//	This is a singleton class, it contains the d3d resource which must be recreated if device is lost
class D3DResource
{
//public method
public:
	//destructor
	~D3DResource();

	//get the single ton
	static D3DResource*	GetSingleton();

	//release the singleton
	static void	Destroy();

	//Load content
	void	LoadContent();

	//Unload content
	void	ReleaseContent( bool releaseTexture );

	//push filename into the texture list
	void	PushTexture( const char* filename );
	//set device
	void	SetDevice( LPDIRECT3DDEVICE9 device );
	//Get shader effect
	LPD3DXEFFECT	GetEffect( int index );

	//get texture
	LPDIRECT3DTEXTURE9	GetTexture( const char* filename );
	//get the index of the texture
	UINT		GetTextureIndex( const char* filename );
	//Get the device of d3d
	LPDIRECT3DDEVICE9	GetDevice();

	//get custom texture
	_float4*	GetCustomTexture();
	//get texture offset
	int*	GetTextureOffset();

	//get cuda texture
	float4*	GetCUDATexture();
	//get cuda texture offset
	int*	GetCUDATextureOffset();

	//whether the resource is ready
	bool	ResourceReady();
	
//private field
private:
	//the only pointer to the instance of the class
	static D3DResource*		m_pSingleton;

	//the device for d3d
	LPDIRECT3DDEVICE9	m_lpd3dDevice;

	//the list of the name for the textures
	vector<string>		m_TextureList;

	//the shader effect
	vector<LPD3DXEFFECT>	m_Effects;
	//the textures
	vector<LPDIRECT3DTEXTURE9>	m_Textures;
	//the offset of custom texture
	int*			m_pTextureOffset;
	//the custom texture for ray tracing
	_float4*		m_pCustomTexture;
	//the cuda memory
	float4*			m_cCustomTexture;
	//cuda texture offset
	int*			m_cTextureOffset;

	//constructor
	D3DResource();

	//Load the default shader effect
	void	LoadDefaultShader();
	//Load the textures
	void	LoadTexture();
	//Release default shader
	void	ReleaseDefaultShader();
	//Release texture
	void	ReleaseTexture();
	//copy the memory to a custom memory
	_float4*	CopyTextureMemory(LPDIRECT3DTEXTURE9 texture);
};