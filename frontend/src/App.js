import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import RAGS from './components/RAG/RAGS';

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/*" element={<RAGS />} />
            </Routes>
        </Router>
    );
}

export default App;
