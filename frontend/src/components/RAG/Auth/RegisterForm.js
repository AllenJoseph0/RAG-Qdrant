import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Eye, EyeOff } from 'lucide-react';
import { RAG_BACKEND_URL } from '../rag.utils';
import './Auth.css';

const RegisterForm = ({ onSwitchToLogin }) => {
    const [registerForm, setRegisterForm] = useState({
        name: '', username: '', password: '', email: '',
        firmId: '', role: 'BUSINESS_USER', designation: '', company: ''
    });
    const [registerError, setRegisterError] = useState('');
    const [registerSuccess, setRegisterSuccess] = useState('');
    const [firms, setFirms] = useState([]);
    const [showPassword, setShowPassword] = useState(false);

    useEffect(() => {
        const fetchFirms = async () => {
            try {
                const res = await axios.get(`${RAG_BACKEND_URL}/api/firms`);
                setFirms(res.data);
            } catch (err) {
                console.error("Failed to fetch firms", err);
            }
        };
        fetchFirms();
    }, []);

    const handleRegisterSubmit = async (e) => {
        e.preventDefault();
        setRegisterError('');
        setRegisterSuccess('');

        if (!registerForm.username || !registerForm.password || !registerForm.name || !registerForm.firmId) {
            setRegisterError("Please fill in all required fields.");
            return;
        }

        try {
            const res = await axios.post(`${RAG_BACKEND_URL}/api/register`, registerForm);
            if (res.data.success) {
                setRegisterSuccess(res.data.message || "Registration successful! Please login.");
                setTimeout(() => {
                    onSwitchToLogin();
                }, 2000);
            }
        } catch (err) {
            console.error("Registration failed", err);
            setRegisterError(err.response?.data?.error || "Registration failed. Please try again.");
        }
    };

    return (
        <>
            <form onSubmit={handleRegisterSubmit} className="auth-form">
                <div>
                    <label className="auth-label">Full Name *</label>
                    <input type="text" value={registerForm.name} onChange={e => setRegisterForm({ ...registerForm, name: e.target.value })} className="auth-input" required />
                </div>
                <div>
                    <label className="auth-label">Username *</label>
                    <input type="text" value={registerForm.username} onChange={e => setRegisterForm({ ...registerForm, username: e.target.value })} className="auth-input" required />
                </div>
                <div>
                    <label className="auth-label">Password *</label>
                    <div className="auth-input-group">
                        <input
                            type={showPassword ? "text" : "password"}
                            value={registerForm.password}
                            onChange={e => setRegisterForm({ ...registerForm, password: e.target.value })}
                            className="auth-input"
                            style={{ paddingRight: '40px' }}
                            required
                            placeholder="Create password"
                        />
                        <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="auth-password-toggle"
                        >
                            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                        </button>
                    </div>
                </div>
                <div>
                    <label className="auth-label">Email</label>
                    <input type="email" value={registerForm.email} onChange={e => setRegisterForm({ ...registerForm, email: e.target.value })} className="auth-input" />
                </div>
                <div style={{ display: 'flex', gap: '1rem' }}>
                    <div style={{ flex: 1 }}>
                        <label className="auth-label">Firm (Company) *</label>
                        <select
                            value={registerForm.firmId}
                            onChange={e => {
                                const selectedFirm = firms.find(f => f.FIRM_ID.toString() === e.target.value);
                                setRegisterForm({
                                    ...registerForm,
                                    firmId: e.target.value,
                                    company: selectedFirm ? selectedFirm.FIRM_NAME : ''
                                });
                            }}
                            className="auth-input"
                            required
                        >
                            <option value="">Select Company...</option>
                            {firms.map(firm => (
                                <option key={firm.FIRM_ID} value={firm.FIRM_ID}>
                                    {firm.FIRM_NAME}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div style={{ flex: 1 }}>
                        <label className="auth-label">Type</label>
                        <select value={registerForm.role} onChange={e => setRegisterForm({ ...registerForm, role: e.target.value })} className="auth-input">
                            <option value="BUSINESS_USER">Business User</option>
                            <option value="ADMIN">Company Admin</option>
                        </select>
                    </div>
                </div>
                <div>
                    <label className="auth-label">Designation</label>
                    <input type="text" value={registerForm.designation} onChange={e => setRegisterForm({ ...registerForm, designation: e.target.value })} className="auth-input" />
                </div>
                <div>
                    <label className="auth-label">Company Name</label>
                    <input type="text" value={registerForm.company} readOnly className="auth-input" style={{ opacity: 0.7, cursor: 'not-allowed' }} />
                </div>

                {registerError && <div className="auth-error" style={{ borderColor: 'transparent', background: 'none', color: '#ef4444', padding: 0 }}>{registerError}</div>}
                {registerSuccess && <div className="auth-success" style={{ textAlign: 'center' }}>{registerSuccess}</div>}

                <button type="submit" className="auth-button-green">
                    Register
                </button>
            </form>
            <div className="auth-footer">
                <button onClick={onSwitchToLogin} className="auth-link">
                    Already have an account? Login
                </button>
            </div>
        </>
    );
};

export default RegisterForm;
