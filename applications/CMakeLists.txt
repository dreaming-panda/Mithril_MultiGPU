# applications
function (add_application application_name)
	add_executable(${application_name} ${application_name}.cc)
	target_link_libraries(${application_name} ${dependencies})
endfunction()

add_application(gcn)
