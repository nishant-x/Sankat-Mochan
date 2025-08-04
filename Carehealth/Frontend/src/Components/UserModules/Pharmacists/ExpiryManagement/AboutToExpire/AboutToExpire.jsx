import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import { Link } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './AboutToExpire.css';

const AboutToExpire = () => {
    const [aboutToExpireProducts, setAboutToExpireProducts] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const navigate = useNavigate(); // Initialize useNavigate

    useEffect(() => {
        fetchAboutToExpireProducts();
    }, []);

    const fetchAboutToExpireProducts = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/inventory/api/about_to_expire');
            if (!response.ok) {
                throw new Error('Failed to fetch products about to expire');
            }
            const data = await response.json();
            setAboutToExpireProducts(data);
            toast.success('Products loaded successfully!');
        } catch (error) {
            setError('Failed to fetch products about to expire.');
            toast.error('Failed to fetch products about to expire.');
            console.error('Error fetching products about to expire:', error);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString) => {
        return dateString ? dateString.split('T')[0] : '';
    };

    return (
        <div className="AboutToExpire-container">
            <ToastContainer />
            <h1 className="AboutToExpire-title">About to Expire Products</h1>
            {loading ? (
                <p className="AboutToExpire-loading">Loading...</p>
            ) : error ? (
                <p className="AboutToExpire-error-message">{error}</p>
            ) : aboutToExpireProducts.length > 0 ? (
                <table className="AboutToExpire-list">
                    <thead>
                        <tr>
                            <th>Item Name</th>
                            <th>Expiry Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        {aboutToExpireProducts.map((product, index) => (
                            <tr key={index} className="AboutToExpire-row">
                                <td>{product.name}</td>
                                <td>{formatDate(product.expiry_date)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            ) : (
                <p className="AboutToExpire-no-products">No products about to expire available.</p>
            )}
            <div className="AboutToExpire-buttons">
                <Link to="/inventory" className="AboutToExpire-back-btn">Go back</Link>
                <Link to="/pharmacy-dashboard" className="AboutToExpire-home-btn">Home</Link>
            </div>
        </div>
    );
};

export default AboutToExpire;
