import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, useLocation, Routes, Route, Navigate } from 'react-router-dom';
import Cookies from 'js-cookie';
import { BrainCircuit } from 'lucide-react';
import styles from './rag.styles.js';
import './RAG.css';

import { RAG_BACKEND_URL } from './rag.utils';
import DashboardPage from './Dashboard/DashboardPage';
import QueryView from './Query/QueryView';
import AuthModal from './Auth/AuthModal';

// ==============================================================================
// Main App Component
// ==============================================================================
const RAGS = () => {
    // view state removed
    const [currentUser, setCurrentUser] = useState(null);
    const navigate = useNavigate();
    const location = useLocation();
    const [showDevLogin, setShowDevLogin] = useState(false);
    const [isDevMode, setIsDevMode] = useState(false);

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
            let role = 'business'; // Default role
            const normalizedType = type ? type.toUpperCase() : '';

            // Map ADMINAPP -> admin
            // Also if cookie says 'SUPER_ADMIN' or 'ADMIN' (though login sets ADMINAPP), handle gracefully
            if (normalizedType === 'ADMINAPP' || normalizedType === 'SUPER_ADMIN' || normalizedType === 'ADMIN') {
                role = 'admin';
            }
            if (normalizedType === 'BUSINESSAPP') role = 'business';

            const user = { id, name, role, firmId };
            setCurrentUser(user);

            axios.post(`${RAG_BACKEND_URL}/api/users/sync`, user).catch(err => console.error("Failed to sync user", err));

            if (role === 'business') {
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

    const handleLoginSuccess = () => {
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
        return <AuthModal onLoginSuccess={handleLoginSuccess} />;
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
