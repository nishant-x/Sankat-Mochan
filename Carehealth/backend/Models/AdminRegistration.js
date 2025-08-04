const mongoose = require('mongoose');

const adminSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
    },
    email: {
        type: String,
        required: true,
    },
    contact: {
        type: String,
        required: true,
        match: [/^[6-9]\d{9}$/, 'Please enter a valid contact number']
    },
    password: {
        type: String,
        required: true,
    },
    userRole: {
        type: String,
        required: true,
    }
    
}, {
    timestamps: true
});



const Admin = mongoose.model('Admin', adminSchema);

module.exports = Admin;
