const express = require("express");
const nodemailer = require("nodemailer");
const router = express.Router();

const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "demofreetouse@gmail.com",
    pass: "ierm wevb owpv fggp",
  },
});

router.post("/send-appointment-confirmation", async (req, res) => {
  const { patientEmail, patientName, appointmentDate, appointmentTime, department } = req.body;

  try {
    const mailOptions = {
      from: "demofreetouse@gmail.com",
      to:  patientEmail,
      subject: "Appointment Confirmation",
      text: `Dear ${patientName},\n\nYour appointment has been successfully booked.\n\nDetails:\nDate: ${appointmentDate}\nTime: ${appointmentTime}\nDepartment: ${department}\n\nThank you for choosing our hospital.`,
      html: `<p>Dear <strong>${patientName}</strong>,</p>
             <p>Your appointment has been successfully booked.</p>
             <p><strong>Details:</strong></p>
             <ul>
               <li><strong>Date:</strong> ${appointmentDate}</li>
               <li><strong>Time:</strong> ${appointmentTime}</li>
               <li><strong>Department:</strong> ${department}</li>
             </ul>
             <p>Thank you for choosing our hospital.</p>`,
    };

    const info = await transporter.sendMail(mailOptions);
    res.status(200).json({ message: "Email sent successfully", info });
  } catch (error) {
    console.error("Error sending email: ", error);
    res.status(500).json({ error: "Failed to send email" });
  }
});

module.exports = router;