# LẬP TRÌNH SONG SONG ỨNG DỤNG - HK2/21-22 - FIT- HCMUS
## REMOVE BACKGROUND USING SUPER PIXEL
### GIẢNG VIÊN HƯỚNG DẪN
Thầy: TRẦN TRUNG KIÊN
### THÀNH VIÊN
1. KIM ĐÌNH LỘC - 1712568
2. NGUYỄN VĂN THÌN - 1712787
3. TRẦN VIỆT VĂN - 1712898
### MỤC TIÊU ĐỒ ÁN
Đồ án hướng đến việc ứng dụng của segmentation trong xử lý ảnh để tách nền các object dựa trên các Super Pixel.
### THÔNG TIN DỰ KIẾN CỦA ĐỒ ÁN
Theo dự kiến, đồ án sẽ thực hiện theo trình tự như sau:
1. Ảnh input đầu vào dạng RGB được convert sang CELAB
2. Đặt các điểm mầm của các Super Pixel.
3. Loops.  
3.1. Tính khoảng cách của các điểm mầm đến các pixel lân cận.  
3.2. Thực hiện gom cụm từng Super Pixel.  
3.3. Thay đổi điểm mầm dựa trên kích thước của cụm.  
4. Sau khi có Super Pixel ta thực hiện Segmetations thông qua một trong các giải pháp như sau: Fully Convolutional Networks for Semantic Segmentation, Unet, Mask-CNN hoặc Mean shift Clustering.
5. Output các ảnh về object mới được segmemtation.
### KẾ HOẠCH THỰC HIỆN
Bảng kế hoạch và phân công công việc: https://trello.com/invite/b/JWBsrVpi/69c8f1fa19a42b8c0be145860aa17b94/applied-parallel-programming
