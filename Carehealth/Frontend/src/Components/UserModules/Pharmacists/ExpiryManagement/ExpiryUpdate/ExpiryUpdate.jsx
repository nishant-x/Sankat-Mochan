import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './ExpiryUpdate.css';

const ExpiryUpdate = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [product, setProduct] = useState(null);
    const [newExpiryDate, setNewExpiryDate] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchProduct();
    }, [id]);

    const fetchProduct = async () => {
        setLoading(true);
        try {
            const response = await fetch(`http://localhost:5000/inventory/api/expired-products/${id}`);
            if (!response.ok) {
                throw new Error('Failed to fetch product details');
            }
            const data = await response.json();
            setProduct(data);
        } catch (error) {
            setError('Failed to fetch product details.');
            toast.error('Failed to fetch product details.');
            console.error('Error fetching product:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleUpdate = async () => {
        try {
            const response = await fetch(`http://localhost:5000/inventory/api/expired-products/${id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ expiry_date: newExpiryDate }),
            });

            if (!response.ok) {
                throw new Error('Failed to update expiry date');
            }

            toast.success('Expiry date updated successfully!');
            setTimeout(() => navigate('/inventory'), 2000);
        } catch (error) {
            setError('Failed to update expiry date.');
            toast.error('Failed to update expiry date.');
            console.error('Error updating expiry date:', error);
        }
    };

    if (loading) return <div className="ExpiryUpdate-loading">Loading details...</div>;
    if (error) return <div className="ExpiryUpdate-error">{error}</div>;
    if (!product) return <div className="ExpiryUpdate-not-found">Product not found!</div>;

    return (
        <div className="ExpiryUpdate-container">
            <ToastContainer />
            <h1 className="ExpiryUpdate-title">Update Expiry Date</h1>
            <p className="ExpiryUpdate-current-date">
                Current Expiry Date: {product.expiry_date ? product.expiry_date.split('T')[0] : 'N/A'}
            </p>
            <input 
                className="ExpiryUpdate-input" 
                type="date" 
                value={newExpiryDate} 
                onChange={(e) => setNewExpiryDate(e.target.value)} 
            />
            <button className="ExpiryUpdate-update-btn" onClick={handleUpdate}>Update</button>
            <button className="ExpiryUpdate-back-btn" onClick={() => navigate('/Expired-product')}>Go Back</button>
        </div>
    );
};

export default ExpiryUpdate;
