import { useState } from 'react'
import './App.css'
import LiveChart from './LiveChart'
// const STOCK = import.meta.env.STOCK


function App() {

  return (
    <>
      <h1>Stock Tracker</h1>
      <LiveChart symbol={'TSLA'}/>
    </>
  )
}

export default App
