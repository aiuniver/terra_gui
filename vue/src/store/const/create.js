const changeStructTable = (data) => {
    const newarr = [];
    data.forEach((el, index) => {
        el.forEach((elm, i) => {
            if (!newarr[i]) {
                newarr[i] = [];
            }
            newarr[i][index] = elm;
        });
    });
    return newarr;
}
const getFiles = (arr) => {
    return arr.map(e => {
        return {
            id: e.id,
            cover: e.cover,
            label: e.title,
            type: e.type,
            table: changeStructTable(e.data || []),
            value: e.path,
        };
    });
}


export { getFiles }