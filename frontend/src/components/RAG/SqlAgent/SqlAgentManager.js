import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { Database, Save, Trash2, RefreshCw, Activity, CheckCircle, AlertTriangle, Layers, Server, ShieldCheck } from 'lucide-react';
import { RAG_BACKEND_URL } from '../rag.utils';
import styles from './sqlAgent.styles';

const SqlAgentManager = ({ currentUser }) => {
    const [configs, setConfigs] = useState([]);
    const [selectedDb, setSelectedDb] = useState(null); // Database Name
    const [viewMode, setViewMode] = useState('list'); // 'list', 'create', 'edit'

    // Form State
    const [formData, setFormData] = useState({
        host: '',
        port: '3306',
        user: '',
        password: '',
        database: ''
    });

    // Status State for Selected DB
    const [status, setStatus] = useState({
        connected: false,
        tables: [],
        classification: 'Unknown',
        generator: 'Unknown'
    });

    const [syncStatus, setSyncStatus] = useState({
        isSyncing: false,
        message: ''
    });

    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [showPassword, setShowPassword] = useState(false);

    // Cache for status to reduce backend calls
    const statusCache = useRef({}); // { dbName: { data: statusObj, timestamp: number } }

    // Initial Fetch
    useEffect(() => {
        fetchConfigs();
    }, [currentUser]);

    // Fetch Status when selected DB changes
    useEffect(() => {
        if (selectedDb && viewMode === 'view') {
            fetchStatus(selectedDb);
        }
    }, [selectedDb, viewMode]);

    const fetchConfigs = async () => {
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/sql_agent/config?user_id=${currentUser?.id}&firm_id=${currentUser?.firmId}`);
            const data = Array.isArray(res.data) ? res.data : (res.data.host ? [res.data] : []);
            setConfigs(data);

            // Auto-select first if none selected and available
            if (data.length > 0 && !selectedDb) {
                setSelectedDb(data[0].database);
                setViewMode('view');
            } else if (data.length === 0) {
                setViewMode('create');
            }
        } catch (err) {
            console.error("Failed to fetch SQL configs", err);
            setError("Failed to load configurations.");
        }
    };

    const fetchStatus = async (dbName, forceRefresh = false) => {
        // Frontend Cache Check
        if (!forceRefresh && statusCache.current[dbName]) {
            const entry = statusCache.current[dbName];
            if (Date.now() - entry.timestamp < 60000) { // 1 minute cache
                setStatus(entry.data);
                return;
            }
        }

        try {
            setLoading(true);
            // We also pass refresh=true to backend if forcing, to bypass backend cache if any
            const res = await axios.get(`${RAG_BACKEND_URL}/api/sql_agent/test?user_id=${currentUser?.id}&firm_id=${currentUser?.firmId}&db_name=${dbName}&refresh=${forceRefresh}`);
            if (res.data) {
                const newStatus = {
                    connected: res.data.success || res.data.connected,
                    tables: res.data.tables || [],
                    classification: res.data.classification_status || 'Ready',
                    generator: res.data.generator_status || 'Ready'
                };
                setStatus(newStatus);
                // Update Cache
                statusCache.current[dbName] = { data: newStatus, timestamp: Date.now() };
            }
        } catch (err) {
            console.error("Failed to fetch SQL status", err);
            setStatus(prev => ({ ...prev, connected: false }));
        } finally {
            setLoading(false);
        }
    };

    const handleEdit = () => {
        const config = configs.find(c => c.database === selectedDb);
        if (config) {
            setFormData({ ...config });
            setViewMode('edit');
            setError('');
            setMessage('');
        }
    };

    const handleCreate = () => {
        setFormData({ host: '', port: '3306', user: '', password: '', database: '' });
        setViewMode('create');
        setError('');
        setMessage('');
    };

    const handleSave = async (e) => {
        e.preventDefault();
        setLoading(true);
        setMessage('');
        setError('');

        try {
            const payload = {
                db_config: { ...formData },
                user_id: currentUser?.id,
                firm_id: currentUser?.firmId
            };
            const res = await axios.post(`${RAG_BACKEND_URL}/api/sql_agent/connect`, payload);
            setMessage(res.data.message || 'Configuration saved.');

            // Refresh configs
            await fetchConfigs();
            setSelectedDb(formData.database);
            setViewMode('view');
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to save configuration.');
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async () => {
        if (!window.confirm(`Are you sure you want to delete configuration for ${selectedDb}?`)) return;
        setLoading(true);
        try {
            await axios.delete(`${RAG_BACKEND_URL}/api/sql_agent/connect`, {
                data: { user_id: currentUser?.id, firm_id: currentUser?.firmId, database: selectedDb }
            });
            setMessage('Configuration deleted.');
            setSelectedDb(null);
            await fetchConfigs(); // Will auto-switch to create or next aval
        } catch (err) {
            setError('Failed to delete configuration.');
        } finally {
            setLoading(false);
        }
    };

    const handleSync = async () => {
        if (!selectedDb) return;
        setSyncStatus({ isSyncing: true, message: 'Starting schema sync...' });
        try {
            await axios.post(`${RAG_BACKEND_URL}/api/sql_agent/sync`, {
                user_id: currentUser?.id,
                firm_id: currentUser?.firmId,
                db_name: selectedDb
            });
            setTimeout(() => {
                setSyncStatus({ isSyncing: false, message: 'Schema synchronization initiated.' });
            }, 2000);
        } catch (err) {
            setSyncStatus({ isSyncing: false, message: 'Failed to start sync.' });
            setError('Schema sync failed to start.');
        }
    };

    const handleCancel = () => {
        if (configs.length > 0) {
            setViewMode('view');
            if (!selectedDb) setSelectedDb(configs[0].database);
        } else {
            // If cancelling create on empty list, stay but maybe clear?
            // Actually if list is empty, we must create.
        }
    };

    return (
        <div className="fade-in">
            {message && <div style={{ ...styles.alert, ...styles.alertSuccess, marginTop: '1rem' }}>{message}</div>}
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}

            <div style={styles.card}>
                <div style={styles.cardHeader}>
                    <Database size={20} /> SQL Agent Configuration
                </div>
                <div style={styles.cardBody}>

                    {/* Database Selector Bar */}
                    <div style={{ display: 'flex', gap: '0.5rem', overflowX: 'auto', paddingBottom: '1rem', borderBottom: '1px solid var(--border)', marginBottom: '1.5rem' }}>
                        {configs.map(c => (
                            <button
                                key={c.database}
                                onClick={() => { setSelectedDb(c.database); setViewMode('view'); }}
                                style={{
                                    padding: '0.5rem 1rem',
                                    borderRadius: '20px',
                                    border: selectedDb === c.database ? '2px solid var(--primary)' : '1px solid var(--border)',
                                    background: selectedDb === c.database ? 'var(--primary-light)' : 'var(--secondary)',
                                    color: selectedDb === c.database ? 'var(--primary)' : 'var(--foreground)',
                                    cursor: 'pointer',
                                    fontWeight: 500,
                                    whiteSpace: 'nowrap',
                                    display: 'flex', alignItems: 'center', gap: '5px'
                                }}
                            >
                                <Database size={14} /> {c.database}
                            </button>
                        ))}
                        <button
                            onClick={handleCreate}
                            style={{
                                padding: '0.5rem 1rem',
                                borderRadius: '20px',
                                border: '1px dashed var(--muted-foreground)',
                                background: 'transparent',
                                color: 'var(--muted-foreground)',
                                cursor: 'pointer',
                                whiteSpace: 'nowrap',
                                display: 'flex', alignItems: 'center', gap: '5px'
                            }}
                        >
                            + Add New
                        </button>
                    </div>

                    {/* CREATE OR EDIT FORM */}
                    {(viewMode === 'create' || viewMode === 'edit') && (
                        <div className="fade-in">
                            <h4 style={{ ...styles.headerSubtitle, marginBottom: '1rem' }}>
                                {viewMode === 'create' ? 'Connect New Database' : `Edit ${formData.database}`}
                            </h4>
                            <form onSubmit={handleSave}>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Host</label>
                                        <input
                                            style={styles.input}
                                            value={formData.host}
                                            onChange={e => setFormData({ ...formData, host: e.target.value })}
                                            placeholder="e.g. 192.168.1.100"
                                            required
                                        />
                                    </div>
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Port</label>
                                        <input
                                            style={styles.input}
                                            value={formData.port}
                                            onChange={e => setFormData({ ...formData, port: e.target.value })}
                                            placeholder="3306"
                                            required
                                        />
                                    </div>
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>User</label>
                                        <input
                                            style={styles.input}
                                            value={formData.user}
                                            onChange={e => setFormData({ ...formData, user: e.target.value })}
                                            required
                                        />
                                    </div>
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Password</label>
                                        <div style={{ position: 'relative' }}>
                                            <input
                                                type={showPassword ? "text" : "password"}
                                                style={styles.input}
                                                value={formData.password}
                                                onChange={e => setFormData({ ...formData, password: e.target.value })}
                                                required
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowPassword(!showPassword)}
                                                style={{
                                                    position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)',
                                                    border: 'none', background: 'none', cursor: 'pointer', color: 'var(--muted-foreground)'
                                                }}
                                            >
                                                {showPassword ? 'Hide' : 'Show'}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                <div style={{ marginTop: '1rem' }}>
                                    <div style={styles.formGroup}>
                                        <label style={styles.label}>Database Name</label>
                                        <input
                                            style={styles.input}
                                            value={formData.database}
                                            onChange={e => setFormData({ ...formData, database: e.target.value })}
                                            required
                                        />
                                    </div>
                                </div>
                                <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem' }}>
                                    <button type="submit" style={styles.buttonSuccess} disabled={loading}>
                                        <Save size={16} /> Save Connection
                                    </button>
                                    {configs.length > 0 && (
                                        <button type="button" onClick={handleCancel} style={styles.buttonSecondary} disabled={loading}>
                                            Cancel
                                        </button>
                                    )}
                                </div>
                            </form>
                        </div>
                    )}

                    {/* VIEW MODE: STATUS & ACTIONS */}
                    {viewMode === 'view' && selectedDb && (
                        <div className="fade-in">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    {status.connected ?
                                        <><span style={{ height: '10px', width: '10px', borderRadius: '50%', background: 'var(--success)' }}></span> <span style={{ fontWeight: 'bold' }}>Connected</span></> :
                                        <><span style={{ height: '10px', width: '10px', borderRadius: '50%', background: 'var(--danger)' }}></span> <span style={{ fontWeight: 'bold' }}>Disconnected</span></>
                                    }
                                </div>
                                <div style={{ display: 'flex', gap: '0.5rem' }}>
                                    <button onClick={handleEdit} style={styles.buttonSecondary} title="Edit Configuration"><Server size={14} /> Edit</button>
                                    <button onClick={handleDelete} style={styles.buttonDangerOutline} title="Delete Configuration"><Trash2 size={14} /></button>
                                </div>
                            </div>

                            {status.connected && (
                                <div style={{ background: 'var(--secondary)', padding: '1rem', borderRadius: 'var(--radius)', marginTop: '1rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                        <strong style={{ display: 'flex', alignItems: 'center', gap: '5px' }}><Layers size={16} /> Available Tables ({status.tables.length})</strong>
                                        <button onClick={() => fetchStatus(selectedDb, true)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--primary)' }}><RefreshCw size={14} /></button>
                                    </div>

                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.75rem', maxHeight: '500px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                                        {status.tables.map((t, idx) => {
                                            const tableName = typeof t === 'object' ? t.table_name : t;
                                            const colCount = typeof t === 'object' ? (t.column_count || 0) : null;
                                            return (
                                                <div key={tableName || idx} style={{
                                                    fontSize: '0.85rem', color: 'var(--foreground)', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                                    padding: '0.5rem 0.75rem', borderRadius: '6px', border: '1px solid var(--border)', backgroundColor: 'var(--background)'
                                                }} title={tableName}>
                                                    <div style={{ display: 'flex', alignItems: 'center', overflow: 'hidden', marginRight: '8px' }}>
                                                        <Layers size={14} style={{ marginRight: '8px', minWidth: '14px', flexShrink: 0, color: 'var(--primary)' }} />
                                                        <span style={{ fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{tableName}</span>
                                                    </div>
                                                    {colCount !== null && (
                                                        <span style={{ fontSize: '0.75rem', color: 'var(--muted-foreground)', padding: '2px 6px', borderRadius: '4px', backgroundColor: 'var(--secondary)', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                                            {colCount} cols
                                                        </span>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            <div style={{ marginTop: '2rem', padding: '1.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)', background: 'var(--card-bg)' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
                                    <RefreshCw size={20} className={syncStatus.isSyncing ? "spin" : ""} />
                                    <h4 style={{ margin: 0 }}>Knowledge Base Sync</h4>
                                </div>
                                <p style={{ fontSize: '0.9rem', color: 'var(--muted-foreground)', marginBottom: '1rem' }}>
                                    Sync <strong>{selectedDb}</strong> schema to the Knowledge Base to allow AI to query it.
                                </p>
                                <button
                                    onClick={handleSync}
                                    style={{ ...styles.buttonPrimary, width: '100%' }}
                                    disabled={syncStatus.isSyncing || !status.connected}
                                >
                                    {syncStatus.isSyncing ? 'Syncing Schema...' : 'Sync Schema to Knowledge Base'}
                                </button>
                                {syncStatus.message && <p style={{ marginTop: '0.5rem', textAlign: 'center', fontSize: '0.9rem', color: 'var(--muted-foreground)' }}>{syncStatus.message}</p>}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SqlAgentManager;
