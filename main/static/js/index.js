$(document).ready(function(){
    $('.artnavig ').click(function(){
        console.log("Center");
        window.location.href = './' + $(this).attr('id')
    })
})