"use strict";


(($) => {


    $(() => {

        $(".custom-scrollbar-wrapper").mCustomScrollbar({
            axis:"y",
            scrollbarPosition:"outside",
            scrollInertia:0,
        });

        $("select.jquery-ui-menuselect").selectmenu();

        $(window).bind("resize", (event) => {
            $("select.jquery-ui-menuselect").selectmenu("close");
        });

    });


})(jQuery);
