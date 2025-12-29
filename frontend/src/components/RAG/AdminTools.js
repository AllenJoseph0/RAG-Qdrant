import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import Cookies from 'js-cookie';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Loader2, TestTube2, FileEdit, Save, ShieldCheck, BrainCircuit, ShieldAlert, PlusSquare, Trash2, Key, X, Share2, Users, Folder, UserX } from 'lucide-react';
import styles from './rag.styles.js';
import { RAG_BACKEND_URL } from './rag.utils';

// ==============================================================================
// Compliance Tester Component
// ==============================================================================
export const ComplianceTester = ({ adminId, categoryName, personaId, complianceProfileId }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [numQuestions, setNumQuestions] = useState(10);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [testResults, setTestResults] = useState(null);
    const firmid = Cookies.get('firmid');

    const handleRunTest = async () => {
        setIsLoading(true);
        setError('');
        setTestResults(null);
        try {
            const payload = {
                owner_id: adminId, // AI Server expects owner_id
                firmId: firmid,
                category: categoryName,
                personaId,
                complianceProfileId,
                num_questions: parseInt(numQuestions, 10) || 10,
            };
            const res = await axios.post(`${RAG_BACKEND_URL}/api/rag/run-test`, payload);
            setTestResults(res.data.results);
        } catch (err) {
            setError(err.response?.data?.error || 'An unexpected error occurred during the test.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={styles.rulebookContainer}>
            <button onClick={() => setIsOpen(!isOpen)} style={styles.rulebookToggleButton}>
                <TestTube2 size={18} /> {isOpen ? 'Hide' : 'Show'} Compliance Q&A Test
            </button>
            {isOpen && (
                <div style={styles.rulebookEditor}>
                    <div style={styles.complianceTestControls}>
                        <div style={styles.formGroup} className="form-group-inline">
                            <label htmlFor={`num-questions-${categoryName}`} style={{ ...styles.label, marginBottom: 0 }}>Number of Questions to Generate:</label>
                            <input
                                id={`num-questions-${categoryName}`}
                                type="number"
                                value={numQuestions}
                                onChange={(e) => setNumQuestions(e.target.value)}
                                style={{ ...styles.input, width: '100px' }}
                                min="1"
                                max="50"
                                disabled={isLoading}
                            />
                        </div>
                        <button onClick={handleRunTest} disabled={isLoading} style={styles.buttonPrimary}>
                            {isLoading ? <><Loader2 size={16} style={styles.spinner} /> Generating & Running...</> : 'Generate & Run Test'}
                        </button>
                    </div>
                    {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}

                    {isLoading && <div style={{ ...styles.loadingContainer, padding: '2rem' }}><Loader2 style={styles.spinner} size={24} /> Running test... this may take a moment.</div>}

                    {testResults && (
                        <div style={styles.testResultsContainer}>
                            <h3 style={styles.testResultsHeader}>Test Results</h3>
                            {testResults.map((result, index) => (
                                <div key={index} style={styles.testResultItem}>
                                    <p style={styles.testResultQuestion}><strong>Q:</strong> {result.question}</p>
                                    <div style={styles.testResultAnswer}>
                                        <strong>A:</strong> <ReactMarkdown children={result.answer} remarkPlugins={[remarkGfm]} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// ==============================================================================
// Redaction Rulebook Component
// ==============================================================================
export const RedactionRulebook = ({ adminId, categoryName }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [rulebookContent, setRulebookContent] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [isSaving, setIsSaving] = useState(false);

    const fetchRulebook = useCallback(async () => {
        setIsLoading(true);
        setError('');
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/rag/rulebook/${adminId}/${categoryName}`);
            setRulebookContent(res.data.content || '');
        } catch (err) {
            setError('Could not load rulebook.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    }, [adminId, categoryName]);

    useEffect(() => {
        if (isOpen) {
            fetchRulebook();
        }
    }, [isOpen, fetchRulebook]);

    const handleSave = async () => {
        setIsSaving(true);
        setError('');
        setSuccess('');
        try {
            await axios.post(`${RAG_BACKEND_URL}/api/rag/rulebook`, {
                adminId,
                category: categoryName,
                rulebookContent: rulebookContent
            });
            setSuccess('Rulebook saved successfully!');
            setTimeout(() => setSuccess(''), 3000);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to save rulebook.');
            console.error(err);
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div style={styles.rulebookContainer}>
            <button onClick={() => setIsOpen(!isOpen)} style={styles.rulebookToggleButton}>
                <FileEdit size={18} /> {isOpen ? 'Hide' : 'Show'} Redaction Rulebook
            </button>
            {isOpen && (
                <div style={styles.rulebookEditor}>
                    {isLoading && <p>Loading rules...</p>}
                    {!isLoading && (
                        <>
                            <div style={styles.rulebookHelpText}>
                                <p style={{ margin: 0, fontWeight: 'bold' }}>How to add rules:</p>
                                <p style={{ margin: '0.5rem 0 0 0' }}>Enter sensitive patterns (regex) to be redacted from the AI's final response, one per line.</p>
                            </div>
                            <textarea
                                value={rulebookContent}
                                onChange={(e) => setRulebookContent(e.target.value)}
                                style={styles.rulebookTextarea}
                                rows={10}
                                placeholder={`# Example: Find and redact Social Security Numbers\n\\b\\d{3}-\\d{2}-\\d{4}\\b`}
                            />
                            <div style={styles.rulebookFooter}>
                                <button onClick={handleSave} disabled={isSaving} style={styles.buttonSuccess}>
                                    {isSaving ? <><Loader2 size={16} style={styles.spinner} /> Saving...</> : <><Save size={16} /> Save Rules</>}
                                </button>
                                {error && <div style={{ ...styles.alert, ...styles.alertDanger, margin: 0 }}>{error}</div>}
                                {success && <div style={{ ...styles.alert, ...styles.alertSuccess, margin: 0 }}>{success}</div>}
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
};

// ==============================================================================
// Category Access Control Component
// ==============================================================================
export const CategoryAccessControl = ({ categoryName, initialPermissions, adminId, onPermissionsChange, currentUserRole }) => {
    const [permissions, setPermissions] = useState(initialPermissions || { business: false, basic: false });
    const [isOpen, setIsOpen] = useState(false);
    const [error, setError] = useState('');

    const handlePermissionChange = async (role, hasAccess) => {
        const newPermissions = { ...permissions, [role]: hasAccess };
        setPermissions(newPermissions);

        try {
            await axios.put(`${RAG_BACKEND_URL}/api/permissions/category`, {
                adminId,
                category: categoryName,
                roleToUpdate: role,
                hasAccess: hasAccess
            });
            onPermissionsChange(categoryName, newPermissions);
        } catch (err) {
            setError('Failed to update. Please refresh.');
            setPermissions(permissions); // Revert on failure
        }
    };

    if (currentUserRole !== 'admin') {
        return null;
    }

    return (
        <div style={styles.permissionsContainer}>
            <button onClick={() => setIsOpen(!isOpen)} style={styles.rulebookToggleButton}>
                <ShieldCheck size={18} /> {isOpen ? 'Hide' : 'Show'} Manage Access
            </button>
            {isOpen && (
                <>
                    {error && <div style={{ padding: '0 1.25rem', color: 'var(--danger)' }}>{error}</div>}
                    <div style={styles.permissionsBody}>
                        <div style={styles.permissionUserRow}>
                            <span>Visible to Business Users</span>
                            <label className="switch">
                                <input
                                    type="checkbox"
                                    checked={permissions.business}
                                    onChange={(e) => handlePermissionChange('business', e.target.checked)}
                                />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div style={styles.permissionUserRow}>
                            <span>Visible to Basic Users</span>
                            <label className="switch">
                                <input
                                    type="checkbox"
                                    checked={permissions.basic}
                                    onChange={(e) => handlePermissionChange('basic', e.target.checked)}
                                />
                                <span className="slider"></span>
                            </label>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

// ==============================================================================
// Persona Selector (for Admin)
// ==============================================================================
export const PersonaSelector = ({ categoryName, adminId, currentPersonaId, personas, onPersonaChange }) => {
    const [selectedId, setSelectedId] = useState(currentPersonaId || '');
    const [isSaving, setIsSaving] = useState(false);
    const [isOpen, setIsOpen] = useState(false);

    const handleChange = async (e) => {
        const newPersonaId = e.target.value;
        setSelectedId(newPersonaId);
        setIsSaving(true);
        try {
            await axios.put(`${RAG_BACKEND_URL}/api/category/settings`, {
                adminId,
                categoryName,
                settings: { personaId: newPersonaId || null }
            });
            onPersonaChange(categoryName, newPersonaId);
        } catch (err) {
            console.error("Failed to save persona setting", err);
            setSelectedId(currentPersonaId || ''); // Revert on failure
        } finally {
            setIsSaving(false);
        }
    };

    useEffect(() => {
        setSelectedId(currentPersonaId || '');
    }, [currentPersonaId]);

    return (
        <div style={styles.personaSelectorContainer}>
            <button onClick={() => setIsOpen(!isOpen)} style={styles.rulebookToggleButton}>
                <BrainCircuit size={18} /> {isOpen ? 'Hide' : 'Show'} Auto-Assign Persona
            </button>
            {isOpen && (
                <div style={styles.personaSelectorBody}>
                    <label style={{ ...styles.label, marginBottom: 0, flexShrink: 0 }}>Default persona for this knowledge base:</label>
                    <select value={selectedId} onChange={handleChange} style={{ ...styles.input, flexGrow: 1 }}>
                        <option value="">-- Manual Selection by User --</option>
                        {personas.map(p => (
                            <option key={p.id} value={p.id}>{p.name}</option>
                        ))}
                    </select>
                    {isSaving && <Loader2 size={16} style={{ ...styles.spinner, marginLeft: '1rem' }} />}
                </div>
            )}
        </div>
    );
};

// ==============================================================================
// Compliance Selector (for Admin)
// ==============================================================================
export const ComplianceSelector = ({ categoryName, adminId, currentProfileId, profiles, onProfileChange }) => {
    const [selectedId, setSelectedId] = useState(currentProfileId || '');
    const [isSaving, setIsSaving] = useState(false);
    const [isOpen, setIsOpen] = useState(false);

    const handleChange = async (e) => {
        const newProfileId = e.target.value;
        setSelectedId(newProfileId);
        setIsSaving(true);
        try {
            await axios.put(`${RAG_BACKEND_URL}/api/category/settings`, {
                adminId,
                categoryName,
                settings: { complianceProfileId: newProfileId || null }
            });
            onProfileChange(categoryName, newProfileId);
        } catch (err) {
            console.error("Failed to save compliance setting", err);
            setSelectedId(currentProfileId || ''); // Revert on failure
        } finally {
            setIsSaving(false);
        }
    };

    useEffect(() => {
        setSelectedId(currentProfileId || '');
    }, [currentProfileId]);

    return (
        <div style={styles.personaSelectorContainer}>
            <button onClick={() => setIsOpen(!isOpen)} style={styles.rulebookToggleButton}>
                <ShieldAlert size={18} /> {isOpen ? 'Hide' : 'Show'} Assign Compliance Profile
            </button>
            {isOpen && (
                <div style={styles.personaSelectorBody}>
                    <label style={{ ...styles.label, marginBottom: 0, flexShrink: 0 }}>Compliance rules for this knowledge base:</label>
                    <select value={selectedId} onChange={handleChange} style={{ ...styles.input, flexGrow: 1 }}>
                        <option value="">-- No compliance checks --</option>
                        {profiles.map(p => (
                            <option key={p.id} value={p.id}>{p.name}</option>
                        ))}
                    </select>
                    {isSaving && <Loader2 size={16} style={{ ...styles.spinner, marginLeft: '1rem' }} />}
                </div>
            )}
        </div>
    );
};

// ==============================================================================
// API Key Manager Component
// ==============================================================================
export const ApiKeyManager = ({ currentUser }) => {
    const [apiKeys, setApiKeys] = useState([]);
    const [llmProviders, setLlmProviders] = useState([]);
    const [llmProviderTypes, setLlmProviderTypes] = useState([]);
    const [editingKey, setEditingKey] = useState(null); // null or key object for modal

    const [selectedProvider, setSelectedProvider] = useState('');
    const [selectedProviderType, setSelectedProviderType] = useState('');
    const [newApiKey, setNewApiKey] = useState('');

    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const firmid = Cookies.get('firmid');

    const fetchLlmOptions = useCallback(async () => {
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/llm/options`);
            setLlmProviders(res.data.providers || []);
            setLlmProviderTypes(res.data.types || []);
            if (res.data.providers && res.data.providers.length > 0) {
                setSelectedProvider(res.data.providers[0]);
            }
            if (res.data.types && res.data.types.length > 0) {
                setSelectedProviderType(res.data.types[0]);
            }
        } catch (err) {
            setError('Failed to fetch LLM provider options.');
        }
    }, []);

    const fetchApiKeys = useCallback(async () => {
        setIsLoading(true);
        if (!firmid || !currentUser.id) return;
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/llm/keys?userId=${currentUser.id}&firmId=${firmid}`);
            setApiKeys(res.data || []);
        } catch (err) {
            setError('Failed to fetch API keys.');
        } finally {
            setIsLoading(false);
        }
    }, [currentUser.id, firmid]);

    useEffect(() => {
        fetchLlmOptions();
        fetchApiKeys();
    }, [fetchLlmOptions, fetchApiKeys]);

    const handleSaveKey = async (e) => {
        e.preventDefault();
        if (!selectedProvider || !selectedProviderType || !newApiKey.trim()) {
            setError('Please fill all fields.');
            return;
        }
        setIsSaving(true);
        setError('');
        setSuccess('');

        try {
            const payload = {
                userId: currentUser.id,
                firmId: firmid,
                llmProvider: selectedProvider,
                llmProviderType: selectedProviderType,
                apiKey: newApiKey.trim()
            };
            await axios.post(`${RAG_BACKEND_URL}/api/llm/keys`, payload);
            setSuccess('API Key saved successfully!');
            setNewApiKey('');
            fetchApiKeys();
            setTimeout(() => setSuccess(''), 3000);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to save API key.');
        } finally {
            setIsSaving(false);
        }
    };

    const handleDeleteKey = async (keyId) => {
        if (!window.confirm('Are you sure you want to delete this API key?')) return;
        try {
            await axios.delete(`${RAG_BACKEND_URL}/api/llm/keys/${keyId}`, {
                data: { userId: currentUser.id, firmId: firmid }
            });
            setSuccess('API Key deleted successfully!');
            fetchApiKeys();
            setTimeout(() => setSuccess(''), 3000);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to delete API key.');
        }
    };

    const handleUpdateKey = async (keyData) => {
        try {
            await axios.put(`${RAG_BACKEND_URL}/api/llm/keys/${keyData.ID}`, {
                ...keyData,
                userId: currentUser.id,
                firmId: firmid
            });
            setSuccess('API Key updated!');
            setEditingKey(null);
            fetchApiKeys();
            setTimeout(() => setSuccess(''), 3000);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to update API key.');
        }
    };

    const maskApiKey = (key) => {
        if (!key || key.length < 8) return '********';
        return `${key.substring(0, 4)}...${key.substring(key.length - 4)}`;
    };

    return (
        <div className="fade-in">
            {success && <div style={{ ...styles.alert, ...styles.alertSuccess, marginTop: '1rem', marginBottom: '1rem' }}>{success}</div>}
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem', marginBottom: '1rem' }}>{error}</div>}

            <div style={styles.card}>
                <div style={styles.cardHeader}><PlusSquare size={20} /> Add New API Key</div>
                <form onSubmit={handleSaveKey}>
                    <div style={styles.cardBody}>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>Provider Type</label>
                            <select style={styles.input} value={selectedProviderType} onChange={e => setSelectedProviderType(e.target.value)} required >
                                <option value="" disabled>Select a type...</option>
                                {llmProviderTypes.map(type => <option key={type} value={type}>{type}</option>)}
                            </select>
                        </div>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>Provider</label>
                            <select style={styles.input} value={selectedProvider} onChange={e => setSelectedProvider(e.target.value)} required >
                                <option value="" disabled>Select a provider...</option>
                                {llmProviders.map(provider => <option key={provider} value={provider}>{provider}</option>)}
                            </select>
                        </div>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>API Key</label>
                            <input type="password" style={styles.input} value={newApiKey} onChange={e => setNewApiKey(e.target.value)} placeholder="Enter your API key" required />
                        </div>
                    </div>
                    <div style={styles.cardFooter}>
                        <button type="submit" style={styles.buttonPrimary} disabled={isSaving}>
                            {isSaving ? <><Loader2 style={styles.spinner} size={16} /> Saving...</> : <><Save size={16} /> Save API Key</>}
                        </button>
                    </div>
                </form>
            </div>

            <div style={styles.card}>
                <div style={styles.cardHeader}><Key size={20} /> Your Saved API Keys</div>
                {isLoading ? (
                    <div style={styles.loadingContainer}><Loader2 style={styles.spinner} size={24} /> Loading keys...</div>
                ) : apiKeys.length === 0 ? (
                    <p style={styles.p}>No API keys saved yet. Add one using the form above.</p>
                ) : (
                    apiKeys.map(key => (
                        <div key={key.ID} style={styles.personaItem}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                <span style={{ ...styles.statusIndicator, backgroundColor: key.STATUS === 'ACTIVE' ? 'var(--success)' : 'var(--muted)' }} title={`Status: ${key.STATUS}`}></span>
                                <div>
                                    <h3 style={styles.personaName}>{key.LLM_PROVIDER} <span style={styles.apiKeyTypeChip}>{key.LLM_PROVIDER_TYPE}</span></h3>
                                    <p style={{ ...styles.personaPrompt, fontFamily: 'monospace' }}>{maskApiKey(key.API_KEY)}</p>
                                </div>
                            </div>
                            <div style={styles.personaActions}>
                                <button onClick={() => setEditingKey(key)} style={styles.buttonSecondary}><FileEdit size={16} /> Edit</button>
                                <button onClick={() => handleDeleteKey(key.ID)} style={styles.buttonDangerOutline}><Trash2 size={16} /> Delete</button>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {editingKey && <ApiKeyEditModal keyData={editingKey} onSave={handleUpdateKey} onCancel={() => setEditingKey(null)} />}
        </div>
    );
};

export const ApiKeyEditModal = ({ keyData, onSave, onCancel }) => {
    const [apiKey, setApiKey] = useState(keyData.API_KEY);
    const [status, setStatus] = useState(keyData.STATUS);
    const [isSaving, setIsSaving] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSaving(true);
        await onSave({ ...keyData, API_KEY: apiKey, STATUS: status });
        setIsSaving(false);
    };

    return (
        <div style={styles.modalOverlay}>
            <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
                <div style={styles.modalHeader}>
                    <h2 style={styles.modalTitle}>Edit API Key for {keyData.LLM_PROVIDER}</h2>
                    <button onClick={onCancel} style={styles.modalCloseButton}><X size={20} /></button>
                </div>
                <form onSubmit={handleSubmit} style={styles.modalBody}>
                    <div style={styles.formGroup}>
                        <label style={styles.label}>API Key (leave unchanged to keep current key)</label>
                        <input className="modal-input" type="password" style={styles.input} value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Enter new key or leave..." />
                    </div>
                    <div style={styles.formGroup}>
                        <label style={styles.label}>Status</label>
                        <select className="modal-input" style={styles.input} value={status} onChange={e => setStatus(e.target.value)}>
                            <option value="ACTIVE">ACTIVE</option>
                            <option value="INACTIVE">INACTIVE</option>
                        </select>
                    </div>
                </form>
                <div style={styles.modalFooter}>
                    <button type="button" onClick={onCancel} style={styles.buttonSecondary}>Cancel</button>
                    <button type="submit" onClick={handleSubmit} style={styles.buttonSuccess} disabled={isSaving}>
                        {isSaving ? <><Loader2 style={styles.spinner} size={16} /> Saving...</> : <><Save size={16} /> Update Key</>}
                    </button>
                </div>
            </div>
        </div>
    );
};

// ==============================================================================
// Share RAG Manager Component
// ==============================================================================
export const SharingManager = ({ currentUser }) => {
    const [employees, setEmployees] = useState([]);
    const [allShares, setAllShares] = useState({}); // { granteeId: [{ownerId, categoryName}, ...], ... }
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const firmid = Cookies.get('firmid');

    const fetchData = useCallback(async () => {
        setIsLoading(true);
        try {
            const fid = currentUser.firmId || Cookies.get('firmid');
            const [empRes, sharesRes] = await Promise.all([
                axios.get(`${RAG_BACKEND_URL}/api/employees?excludeId=${currentUser.id}`),
                axios.get(`${RAG_BACKEND_URL}/api/rag/shares/${fid}`)
            ]);
            setEmployees(empRes.data || []);
            setAllShares(sharesRes.data || {});
        } catch (err) {
            setError("Failed to load sharing data.");
        } finally {
            setIsLoading(false);
        }
    }, [currentUser.id]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    const handleRevoke = async (granteeId, categoryName) => {
        if (!window.confirm(`Are you sure you want to revoke access to '${categoryName}' for this user?`)) return;

        const fid = currentUser.firmId || Cookies.get('firmid');
        setError('');
        setSuccess('');
        try {
            await axios.delete(`${RAG_BACKEND_URL}/api/rag/share`, {
                data: {
                    ownerId: fid,
                    categoryName: categoryName,
                    granteeId: granteeId
                }
            });
            setSuccess('Access revoked successfully.');
            fetchData(); // Refetch all data
            setTimeout(() => setSuccess(''), 4000);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to revoke access.');
        }
    };

    if (isLoading) {
        return <div style={styles.loadingContainer}><Loader2 style={styles.spinner} /> Loading...</div>;
    }

    const employeeMap = employees.reduce((acc, emp) => {
        acc[emp.EMPID] = emp;
        return acc;
    }, {});

    const sharesByEmployee = Object.entries(allShares).map(([granteeId, shares]) => ({
        employee: employeeMap[granteeId] || { EMPID: granteeId, EMPNAME: `Unknown User (${granteeId})` },
        shares: shares
    }));


    return (
        <div className="fade-in">
            {success && <div style={{ ...styles.alert, ...styles.alertSuccess, marginTop: '1rem' }}>{success}</div>}
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}

            <GrantAccessForm currentUser={currentUser} employees={employees} allShares={allShares} onShare={fetchData} />

            <div style={styles.card}>
                <div style={styles.cardHeader}><Users size={20} /> Current Sharing Permissions</div>
                <div style={styles.cardBody}>
                    {sharesByEmployee.length === 0 ? (
                        <p style={{ textAlign: 'left', padding: 0 }}>You have not shared any knowledge bases.</p>
                    ) : (
                        sharesByEmployee.map(({ employee, shares }) => (
                            <div key={employee.EMPID} style={styles.shareGroup}>
                                <h3 style={styles.shareGroupTitle}>{employee.EMPNAME}</h3>
                                {shares.map(share => (
                                    <div key={`${share.ownerId}-${share.categoryName}`} style={styles.personaItem}>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Folder size={16} /> {share.categoryName}</span>
                                        <button onClick={() => handleRevoke(employee.EMPID, share.categoryName)} style={styles.buttonDangerOutline}><UserX size={16} /> Revoke</button>
                                    </div>
                                ))}
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export const GrantAccessForm = ({ currentUser, employees, allShares, onShare }) => {
    const [categories, setCategories] = useState([]);
    const [selectedCategory, setSelectedCategory] = useState('');
    const [selectedEmployee, setSelectedEmployee] = useState('');
    const [isSharing, setIsSharing] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    useEffect(() => {
        const fid = currentUser.firmId || Cookies.get('firmid');
        if (!fid) return;
        axios.get(`${RAG_BACKEND_URL}/api/rag/structure?username=${fid}`)
            .then(res => {
                setCategories(res.data?.[fid] || []);
            })
            .catch(() => setError("Could not load your knowledge bases."));
    }, [currentUser.firmId]);

    const availableEmployees = useMemo(() => {
        if (!selectedCategory || !allShares || !employees) {
            return employees || [];
        }

        const granteesWithAccess = new Set(
            Object.keys(allShares).filter(granteeId =>
                allShares[granteeId].some(share => share.categoryName === selectedCategory)
            )
        );

        return employees.filter(emp => !granteesWithAccess.has(String(emp.EMPID)));
    }, [selectedCategory, employees, allShares]);

    useEffect(() => {
        // When the list of available employees changes (e.g., after selecting a different KB),
        // check if the currently selected employee is still valid. If not, reset it.
        if (selectedEmployee && !availableEmployees.some(e => String(e.EMPID) === selectedEmployee)) {
            setSelectedEmployee('');
        }
    }, [availableEmployees, selectedEmployee]);

    const handleShare = async (e) => {
        e.preventDefault();
        if (!selectedCategory || !selectedEmployee) {
            setError('Please select a knowledge base and an employee.');
            return;
        }
        setIsSharing(true);
        setError('');
        setSuccess('');
        try {
            const payload = {
                ownerId: currentUser.firmId || Cookies.get('firmid'),
                categoryName: selectedCategory,
                granteeId: selectedEmployee
            };
            await axios.post(`${RAG_BACKEND_URL}/api/rag/share`, payload);
            setSuccess(`Successfully shared '${selectedCategory}' with the selected employee.`);
            setSelectedEmployee(''); // Reset employee dropdown
            onShare(); // Callback to refresh parent
            setTimeout(() => setSuccess(''), 4000);
        } catch (err) {
            if (err.response && (err.response.status === 409 || err.response.status === 400)) {
                setError(err.response.data.error);
            } else {
                setError('An unexpected error occurred while sharing. Please try again.');
            }
            console.error('Failed to share RAG', err);
        } finally {
            setIsSharing(false);
        }
    };

    return (
        <>
            {success && <div style={{ ...styles.alert, ...styles.alertSuccess, marginBottom: '1rem' }}>{success}</div>}
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginBottom: '1rem' }}>{error}</div>}
            <div style={styles.card}>
                <div style={styles.cardHeader}><Share2 size={20} /> Grant Access to a Knowledge Base</div>
                <form onSubmit={handleShare}>
                    <div style={styles.cardBody}>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>1. Select Knowledge Base</label>
                            <select style={styles.input} value={selectedCategory} onChange={e => setSelectedCategory(e.target.value)} required>
                                <option value="" disabled>Choose a category...</option>
                                {categories.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                            </select>
                        </div>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>2. Select Employee to Share With</label>
                            <select style={styles.input} value={selectedEmployee} onChange={e => setSelectedEmployee(e.target.value)} required disabled={!selectedCategory}>
                                <option value="" disabled>Choose an employee...</option>
                                {availableEmployees.map(e => <option key={e.EMPID} value={e.EMPID}>{e.EMPNAME} ({e.TYPE || 'No Designation'})</option>)}
                            </select>
                            {selectedCategory && availableEmployees.length === 0 && employees.length > 0 && (
                                <p style={styles.formHelperText}>All employees already have access to this knowledge base.</p>
                            )}
                        </div>
                    </div>
                    <div style={styles.cardFooter}>
                        <button type="submit" style={styles.buttonPrimary} disabled={isSharing || !selectedEmployee}>
                            {isSharing ? <><Loader2 style={styles.spinner} size={16} /> Granting Access...</> : <><Users size={16} /> Grant Access</>}
                        </button>
                    </div>
                </form>
            </div>
        </>
    );
};
