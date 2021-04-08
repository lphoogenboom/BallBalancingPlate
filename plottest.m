figure(1)
stairs(T_1, x(1,:), 'b', 'LineWidth', 1.3);
stairs(T_1, x(3,:), 'r', 'LineWidth', 1.3);
legend('xb', 'xy');

figure(2)
for i = 1:dim.nu
    hold on
    stairs(T, u_rec(i,:), 'LineWidth', 1.3);
    hold off
end
legend('u1', 'u2', 'u3', 'u4');