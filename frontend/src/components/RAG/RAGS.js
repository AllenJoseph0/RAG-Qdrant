import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, useLocation, Routes, Route, Navigate } from 'react-router-dom';
import Cookies from 'js-cookie';
import { BrainCircuit } from 'lucide-react';
import styles from './rag.styles.js';
import './RAG.css';

import { RAG_BACKEND_URL } from './rag.utils';
import DashboardPage from './DashboardPage';
import QueryView from './QueryView';

// ==============================================================================
// Main App Component
// ==============================================================================
const RAGS = () => {
    // view state removed
    const [currentUser, setCurrentUser] = useState(null);
    const navigate = useNavigate();
    const location = useLocation();
    const [showDevLogin, setShowDevLogin] = useState(false);
    // Track if we are in "dev mode" (untrusted device)
    const [isDevMode, setIsDevMode] = useState(false);
    const [devForm, setDevForm] = useState({ name: '', userid: '', firmid: '5', usertype: 'admin' });

    useEffect(() => {
        const footer = document.querySelector("footer");
        if (footer) {
            const updateFooterHeight = () => {
                document.documentElement.style.setProperty(
                    "--footer-height",
                    `${footer.getBoundingClientRect().height}px`
                );
            };
            updateFooterHeight();
            window.addEventListener("resize", updateFooterHeight);
            return () => window.removeEventListener("resize", updateFooterHeight);
        }
    }, []);

    const syncUser = React.useCallback(() => {
        const id = Cookies.get('userid');
        const name = Cookies.get('name');
        const type = Cookies.get('usertype'); // Expects USERAPP, BUSINESSAPP, or ADMINAPP
        const firmId = Cookies.get('firmid');

        if (id && name && type && firmId) {
            let role = 'basic'; // Default role
            if (type === 'ADMINAPP') role = 'admin';
            if (type === 'BUSINESSAPP') role = 'business';

            const user = { id, name, role, firmId };
            setCurrentUser(user);

            axios.post(`${RAG_BACKEND_URL}/api/users/sync`, user).catch(err => console.error("Failed to sync user", err));

            if (role === 'business' || role === 'basic') {
                // Determine if we need to redirect
                if (window.location.pathname.includes('/dashboard')) {
                    navigate('/rag');
                }
            }
        } else {
            navigate('/');
        }
    }, [navigate]);

    useEffect(() => {
        const checkEnvironmentAndSync = async () => {
            try {
                const res = await axios.get(`${RAG_BACKEND_URL}/api/env-check`);
                const isTrustedDevice = res.data.isLocal;
                setIsDevMode(!isTrustedDevice);
                const id = Cookies.get('userid');

                // If NOT trusted (unknown device) and NO user cookies, show popup.
                if (!isTrustedDevice && !id) {
                    setShowDevLogin(true);
                } else {
                    syncUser();
                }
            } catch (error) {
                console.error("Environment check failed", error);
                // Fallback: assume trusted if check fails to avoid locking out, or handle otherwise
                syncUser();
            }
        };
        checkEnvironmentAndSync();
    }, [syncUser]);

    const handleDevLoginSubmit = (e) => {
        e.preventDefault();
        const { name, userid, firmid, usertype } = devForm;
        if (!name || !userid || !firmid) {
            alert("All fields are required.");
            return;
        }

        let cookieType = 'USERAPP';
        if (usertype === 'admin') cookieType = 'ADMINAPP';
        else if (usertype === 'business') cookieType = 'BUSINESSAPP';

        Cookies.set('name', name, { expires: 7 });
        Cookies.set('userid', userid, { expires: 7 });
        Cookies.set('firmid', firmid, { expires: 7 });
        Cookies.set('usertype', cookieType, { expires: 7 });

        setShowDevLogin(false);
        syncUser();
    };

    const handleLogout = () => {
        Cookies.remove('name');
        Cookies.remove('userid');
        Cookies.remove('firmid');
        Cookies.remove('usertype');
        setCurrentUser(null);
        setShowDevLogin(true);
    };

    if (showDevLogin) {
        return (
            <div style={styles.appContainer}>
                <nav style={styles.navbar}>
                    <div style={styles.navLeft}>
                        <BrainCircuit size={28} style={{ color: 'var(--primary)' }} />
                        <h1 style={styles.navTitle}>Cognitive Agent</h1>
                    </div>
                </nav>
                <div style={{
                    position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
                    backgroundColor: 'rgba(0, 0, 0, 0.85)', zIndex: 10000,
                    display: 'flex', justifyContent: 'center', alignItems: 'center',
                    backdropFilter: 'blur(5px)'
                }}>
                    <div style={{
                        backgroundColor: '#1a1a1a', padding: '2rem', borderRadius: '12px',
                        border: '1px solid #333', width: '400px', boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
                        color: '#eee', fontFamily: 'Inter, sans-serif'
                    }}>
                        <h2 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', borderBottom: '1px solid #333', paddingBottom: '1rem' }}>
                            Developer Login
                        </h2>
                        <form onSubmit={handleDevLoginSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: '#aaa' }}>Name</label>
                                <input
                                    type="text"
                                    value={devForm.name}
                                    onChange={e => setDevForm({ ...devForm, name: e.target.value })}
                                    style={{ width: '100%', padding: '0.8rem', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#252525', color: '#fff', fontSize: '1rem' }}
                                    placeholder="Enter your name"
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: '#aaa' }}>User ID</label>
                                <input
                                    type="text"
                                    value={devForm.userid}
                                    onChange={e => setDevForm({ ...devForm, userid: e.target.value })}
                                    style={{ width: '100%', padding: '0.8rem', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#252525', color: '#fff', fontSize: '1rem' }}
                                    placeholder="e.g. 1001"
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: '#aaa' }}>Firm ID</label>
                                <input
                                    type="text"
                                    value={devForm.firmid}
                                    onChange={e => setDevForm({ ...devForm, firmid: e.target.value })}
                                    style={{ width: '100%', padding: '0.8rem', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#252525', color: '#fff', fontSize: '1rem' }}
                                    placeholder="e.g. 5"
                                />
                            </div>
                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: '#aaa' }}>User Type</label>
                                <select
                                    value={devForm.usertype}
                                    onChange={e => setDevForm({ ...devForm, usertype: e.target.value })}
                                    style={{ width: '100%', padding: '0.8rem', borderRadius: '6px', border: '1px solid #333', backgroundColor: '#252525', color: '#fff', fontSize: '1rem' }}
                                >
                                    <option value="basic">Basic (User)</option>
                                    <option value="business">Business</option>
                                    <option value="admin">Admin</option>
                                </select>
                            </div>
                            <button type="submit" style={{
                                marginTop: '1rem', padding: '0.8rem', backgroundColor: '#3b82f6', color: '#fff',
                                border: 'none', borderRadius: '6px', fontSize: '1rem', fontWeight: '600', cursor: 'pointer',
                                transition: 'background 0.2s'
                            }}>
                                Check In
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        );
    }

    if (!currentUser) {
        return <div style={styles.appContainer}><div style={styles.loadingContainer}><h2>Loading User...</h2></div></div>;
    }

    const renderNav = () => currentUser.role === 'admin' ? (
        <div style={styles.navButtonGroup}>
            <button
                onClick={() => navigate('/rag/dashboard')}
                style={location.pathname === '/rag/dashboard' ? styles.navButtonActive : styles.navButton}
            >
                Dashboard
            </button>
            <button
                onClick={() => navigate('/rag')}
                style={location.pathname === '/rag' ? styles.navButtonActive : styles.navButton}
            >
                Query RAG
            </button>
        </div>
    ) : null;

    const renderRoutes = () => (
        <Routes>
            <Route path="/rag/dashboard" element={
                currentUser.role === 'admin' ? <DashboardPage currentUser={currentUser} /> : <Navigate to="/rag" replace />
            } />
            <Route path="/rag" element={<QueryView currentUser={currentUser} />} />
            <Route path="/" element={<Navigate to="/rag" replace />} />
            <Route path="*" element={<Navigate to="/rag" replace />} />
        </Routes>
    );

    return (
        <div style={styles.appContainer}>
            <nav style={styles.navbar}>
                <div style={styles.navLeft}>
                    <BrainCircuit size={28} style={{ color: 'var(--primary)' }} />
                    <h1 style={styles.navTitle}>Cognitive Agent</h1>
                </div>
                <div style={styles.navCenter}>
                    {renderNav()}
                </div>
                <div style={styles.navRight}>
                    <span style={styles.loggedInAs}>Welcome, {currentUser.name}</span>
                    {isDevMode && (
                        <button
                            onClick={handleLogout}
                            style={{
                                marginLeft: '1rem',
                                padding: '0.4rem 0.8rem',
                                backgroundColor: '#b91c1c',
                                color: '#fee2e2',
                                border: '1px solid #991b1b',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '0.85rem',
                                fontWeight: '600',
                                transition: 'background-color 0.2s'
                            }}
                            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#991b1b'}
                            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#b91c1c'}
                        >
                            Logout
                        </button>
                    )}
                </div>
            </nav>
            <main style={styles.mainContent}>{renderRoutes()}</main>
        </div>
    );
};

export default RAGS;
