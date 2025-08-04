import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './ExpiredProduct.css';

const ExpiredProduct = () => {
    const [expiredProducts, setExpiredProducts] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchExpiredProducts();
    }, []);

    const fetchExpiredProducts = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/inventory/api/expired-products');
            if (!response.ok) {
                throw new Error('Failed to fetch expired products');
            }
            const data = await response.json();
            setExpiredProducts(data);
            toast.success('Expired products loaded successfully!');
        } catch (error) {
            setError("There is no expired Medicine's");
            toast.error("Failed to fetch expired products.");
            console.error('Error fetching expired products:', error);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString) => {
        return dateString ? dateString.split('T')[0] : '';
    };

    return (
        <div className="ExpiredProduct-container">
            <ToastContainer />
            <h1 className="ExpiredProduct-heading">Expired Products</h1>
            {loading ? (
                <p className="ExpiredProduct-loading">Loading...</p>
            ) : error ? (
                <p className="ExpiredProduct-error-message">{error}</p>
            ) : expiredProducts.length > 0 ? (
                <table className="ExpiredProduct-list">
                    <thead>
                        <tr>
                            <th>Item Name</th>
                            <th>Expiry Date</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {expiredProducts.map((product, index) => (
                            <tr key={index} className="ExpiredProduct-row">
                                <td className="ExpiredProduct-item-name">{product.name}</td>
                                <td className="ExpiredProduct-expiry-date">{formatDate(product.expiry_date)}</td>
                                <td>
                                    <Link to={`/ExpiryUpdate/${product._id}`} className="ExpiredProduct-action-btn">
                                        Handle
                                    </Link>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            ) : (
                <p className="ExpiredProduct-no-items">No expired products available.</p>
            )}
            <div className="ExpiredProduct-button-container">
                <Link to="/inventory" className="ExpiredProduct-back-btn">Go back</Link>
                <Link to="/pharmacy-dashboard" className="ExpiredProduct-home-btn">Home</Link>
            </div>
        </div>
    );
};

export default ExpiredProduct;
