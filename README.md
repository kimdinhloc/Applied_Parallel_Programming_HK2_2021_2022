## LẬP TRÌNH SONG SONG ỨNG DỤNG - HK2/21-22 - FIT- HCMUS
### REMOVE BACKGROUND USING SUPER PIXEL
### GIẢNG VIÊN HƯỚNG DẪN
Thầy: Trần Trung Kiên - Nguyễn Thái Vũ

### THÀNH VIÊN

| MSSV | Họ tên | Tài khoản Github |
| --- | --- | --- |
| 1712568 | KIM ĐÌNH LỘC | [kimdinhloc](https://github.com/kimdinhloc) |
| 1712787 | NGUYỄN VĂN THÌN | [thinnguyenqb](https://github.com/thinnguyenqb) |
| 1712898 | TRẦN VIỆT VĂN | [tranvietvan](https://github.com/tranvietvan) |

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

<table>
<thead>
  <tr>
    <th>Tuần</th>
    <th>Thời gian</th>
    <th>Lộc</th>
    <th>Thìn</th>
    <th>Văn</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">1</td>
    <td>24/03-29/03</td>
    <td colspan="3">Mỗi người tìm hiểu và đưa ra một đề xuất đề tài cho đồ án cuối kỳ</td>
  </tr>
  <tr>
    <td>30/04</td>
    <td colspan="3">Họp nhóm và thống nhất đề tài</td>
  </tr>
  <tr>
    <td rowspan="2">2</td>
    <td> 31/03-05/04</td>
    <td colspan="3"></td>
  </tr>
  <tr>
    <td>06/04</td>
    <td colspan="3">Chuản bị báo cáo lần 1</td>
  </tr>
  
  <tr>
    <td rowspan="2">3</td>
    <td>07/04-13/04</td>
    <td colspan="3"></tr>
  </tr>
  <tr>
    <td>14/04</td>
    <td colspan="3"></td>
  </tr>
 
  
</tbody>
</table>