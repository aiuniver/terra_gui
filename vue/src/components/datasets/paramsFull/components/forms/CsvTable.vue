<template>
  <div class="csv-table">
    <div
        class="table__col"
        v-for="(row, index) in table_test"
        :key="row"
        @mousedown="select(index)"
        @mouseover="select(index)"
        :class="{ selected: selected_cols.includes(index) }"
    >
      <div
          class="table__row"
          v-for="item in row"
          :key="item"
      >{{ item }}</div>
    </div>
  </div>
</template>

<script>
export default {
  name: "CsvTable",
  data: () => ({
    table_test: [],
    selected_cols: [],
  }),
  created() {
    let file = ";text;text larrrrrge;text normal;text;text;text;text\n" +
        "0;1;text;text;text 2131;NaN;text 2131. dddd;2\n" +
        "2;1;text;text;text 2131;text 2131. dddd;text 2131;2\n" +
        "4;1;text;text;text 2131;text 2131. dddd;text 2131. dddd;2\n" +
        "6;1;text;text;text 2131;text 2131. dddd;text 2131. dddd;2\n" +
        "8;1;text;text;text 2131;NaN;text 2131;2";

    let copy = this.$papa.parse(file).data;
    this.table_test = []

    let row = copy.length;
    let col = copy[0].length

    for(let i = 0; i < col; ++i){
      this.table_test.push([]);
      for(let j = 0; j < row; ++j){

        this.table_test[i].push(copy[j][i] ? copy[j][i] : "-");
      }
    }
    console.log(this.table_test)
  },
  methods: {
    select(index){
      event.preventDefault();
      if(event.which == 1){
        const key = this.selected_cols.indexOf(index);
        if (key !== -1) {
          this.selected_cols.splice(key, 1);
        } else {
          this.selected_cols.push(index);
        }
      }
    }
  },
}
</script>

<style lang="scss" scoped>
.csv-table {
  border-collapse: collapse;
  border: 1px solid #6C7883;
  border-radius: 8px;
  padding: 24px 0 2px 0;
  display: flex;

  .table__col {
    display: flex;
    flex-direction: column;
  }
  .table__row {
    padding: 1px 7px;
    &:nth-child(even) {
      background: #242F3D;
    }
  }
}

.selected{
  border: 1px solid #89D764;
  color:#fff;

  &:nth-child(1){
    border-radius: 6px 0 0 6px;
  }
  &:last-child{
    border-radius: 0 6px 6px 0;
  }
}
</style>