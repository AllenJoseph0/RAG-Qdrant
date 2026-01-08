import React, { useState } from 'react';
import axios from 'axios';
import Cookies from 'js-cookie';
import { Eye, EyeOff } from 'lucide-react';
import { RAG_BACKEND_URL } from '../rag.utils';
import './Auth.css';

const LoginForm = ({ onLoginSuccess, onSwitchToRegister }) => {
    const [loginForm, setLoginForm] = useState({ username: '', password: '' });
    const [loginError, setLoginError] = useState('');
    const [isLoggingIn, setIsLoggingIn] = useState(false);
    const [showPassword, setShowPassword] = useState(false);

    const handleLoginSubmit = async (e) => {
        e.preventDefault();
        setLoginError('');
        setIsLoggingIn(true);

        try {
            const { username, password } = loginForm;
            if (!username || !password) {
                setLoginError("Username and password are required.");
                setIsLoggingIn(false);
                return;
            }

            const res = await axios.post(`${RAG_BACKEND_URL}/api/login`, { username, password });

            if (res.data.success) {
                const { user } = res.data;
                const { id, name, role, firmId } = user;

                let cookieType = 'BUSINESSAPP'; // Default
                if (role === 'admin') cookieType = 'ADMINAPP';

                Cookies.set('name', name, { expires: 7 });
                Cookies.set('userid', id, { expires: 7 });
                Cookies.set('firmid', firmId, { expires: 7 });
                Cookies.set('usertype', cookieType, { expires: 7 });

                if (onLoginSuccess) onLoginSuccess(user);
            }
        } catch (err) {
            console.error("Login Error", err);
            setLoginError(err.response?.data?.error || "Login failed. Please check your credentials.");
        } finally {
            setIsLoggingIn(false);
        }
    };

    return (
        <>
            <form onSubmit={handleLoginSubmit} className="auth-form">
                <div>
                    <label className="auth-label">Username</label>
                    <input
                        type="text"
                        value={loginForm.username}
                        onChange={e => setLoginForm({ ...loginForm, username: e.target.value })}
                        className="auth-input"
                        placeholder="Enter username"
                        autoFocus
                    />
                </div>
                <div>
                    <label className="auth-label">Password</label>
                    <div className="auth-input-group">
                        <input
                            type={showPassword ? "text" : "password"}
                            value={loginForm.password}
                            onChange={e => setLoginForm({ ...loginForm, password: e.target.value })}
                            className="auth-input"
                            placeholder="Enter password"
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

                {loginError && (
                    <div className="auth-error">
                        {loginError}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={isLoggingIn}
                    className="auth-button-primary"
                >
                    {isLoggingIn ? "Authenticating..." : "Sign In"}
                </button>
            </form>
            <div className="auth-footer">
                <button onClick={onSwitchToRegister} className="auth-link">
                    Don't have an account? Create one
                </button>
            </div>
        </>
    );
};

export default LoginForm;
