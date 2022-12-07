

settings_style = '''
        <style>
            table,
            th,
            td {
                border: 1px solid black;
                border-collapse: collapse;
                padding: 5px;
            }

            /* Remove default bullets */
            ul,
            #myUL {
                list-style-type: none;
            }

            /* Remove margins and padding from the parent ul */
            #myUL {
                margin: 0;
                padding: 0;
            }

            /* Style the caret/arrow */
            .caret {
                cursor: pointer;
                user-select: none;
                /* Prevent text selection */
            }

            /* Create the caret/arrow with a unicode, and style it */
            .caret::before {
                color: black;
                display: inline-block;
                margin-right: 6px;
            }

            /* Rotate the caret/arrow icon when clicked on (using JavaScript) */
            .caret-down::before {
                transform: rotate(90deg);
            }

            /* Hide the nested list */
            .nested {
                display: none;
            }

            /* Show the nested list when the user clicks on the caret/arrow (with JavaScript) */
            .active {
                display: block;
            }
        </style>
        '''


start_report = '''
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, minimal-ui">
            <title>Dashboard😱</title>
        </head>
        '''

end_report = '''
    <script type="text/javascript">
        var toggler = document.getElementsByClassName("caret");
        var i;

        for (i = 0; i < toggler.length; i++) {
            toggler[i].addEventListener("click", function () {
                this.parentElement.querySelector(".nested").classList.toggle("active");
                this.classList.toggle("caret-down");
            });
        }
    </script>
    '''