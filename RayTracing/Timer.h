/*
 *	FileName:		Timer.h
 *
 *	Programmer:		Jiayin Cao
 *
 *	Description:	A time counter
 */

#pragma once

/////////////////////////////////////////////////////////////////////////////////
class	Timer
{
//public method
public:
	Timer();
	~Timer();

	//reset the timer
	void	Reset();

	//start the counter
	void	Start();

	//stop the counter
	void	Stop();

	//get elapsed time
	float	GetElapsedTime();

//private field
private:
	//the frequency
	__int64	m_Freq;

	//the clocks
	__int64 m_Clocks;

	//the start time
	__int64	m_Start;
};