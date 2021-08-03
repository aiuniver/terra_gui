<template>
  <div class="csv-table">
    <div class="table__data">
      <div class="table__col">
        <div class="table__row"></div>
        <div class="table__row">0</div>
        <div class="table__row">2</div>
        <div class="table__row">4</div>
        <div class="table__row">6</div>
        <div class="table__row">8</div>
      </div>
      <div class="selected__cols"></div>
      <div
          class="table__col"
          v-for="(row, index) in table_test"
          :key="row"
          @mousedown="select(index)"
          :data-index="index"
      >
        <div
            class="table__row"
            v-for="item in row"
            :key="item"
        >{{ item }}</div>
      </div>
    </div>
    <div class="table__footer"><span>Папка с картинками</span></div>
  </div>
</template>

<script>
export default {
  name: "CardTable",
  data: () => ({
    table_test: [],
    selected_cols: [],
  }),
  created() {
    let file = "text;text larrrrrge;text normal;text;text;text;text\n" +
        "1;text;text;text 2131;NaN;text 2131. dddd;2\n" +
        "1;text;text;text 2131;text 2131. dddd;text 2131;2\n" +
        "1;text;text;text 2131;text 2131. dddd;text 2131. dddd;2\n" +
        "1;text;text;text 2131;text 2131. dddd;text 2131. dddd;2\n" +
        "1;text;text;text 2131;NaN;text 2131;2";

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
  },
  methods: {
    compare(a, b) {
      if (a.dataset.index < b.dataset.index) {
        return -1;
      }
      if (a.dataset.index > b.dataset.index) {
        return 1;
      }
      return 0;
    },
    sortOnDataIndex(el){
      let arr = [], i = el.children.length;
      while(i--){
        arr[i] = el.children[i];
        el.children[i].remove()
      }
      arr.sort(this.compare);
      i = 0;
      while(arr[i]) {
        el.appendChild(arr[i]);
        ++i;
      }
    },
    select(index){
      event.preventDefault();
      if(event.which == 1){
        const key = this.selected_cols.indexOf(index);
        const selected_cols = document.querySelector(".selected__cols");
        const unselected_cols = document.querySelector(".table__data");
        let col = document.querySelector(`.table__col[data-index='${index}']`);
        if (key !== -1) {
          this.selected_cols.splice(key, 1);
          document.querySelector(`.selected__cols`).removeChild(col)
          unselected_cols.append(col);
          this.sortOnDataIndex(unselected_cols);
        } else {
          this.selected_cols.push(index);
          document.querySelector(`.table__data`).removeChild(col)
          selected_cols.append(col);
          this.sortOnDataIndex(selected_cols);
        }

        if(this.selected_cols.length == 0){
          selected_cols.style.display = 'none';
        }else{
          selected_cols.style.display = 'flex';
        }
      }
    }
  },
}
</script>

<style lang="scss" scoped>
.csv-table {
  font-size: 0.75rem;
  border-collapse: collapse;
  border: 1px solid #6C7883;
  border-radius: 8px;
  padding: 23px 0 2px 0;
  display: flex;
  height: 152px;
  flex-direction: column;

  .table__data{
    display: flex;
  }

  .table__col {
    display: flex;
    flex-direction: column;
  }
  .table__row {
    height: 17px;
    padding: 0 8px;
    &:nth-child(even) {
      background: #242F3D;
    }
    &:first-child {
      font-weight: bold;
      height: 19px;
    }
  }

  .table__footer{
    height: 24px;
    width: 100%;
    padding: 3px 8px;
  }
}

.selected__cols{
  display: flex;
  border: 1px solid #89D764;
  border-radius: 4px;
}

.selected{
  border: 1px solid #89D764;
  border-radius: 4px;
  color:#fff;

  &:nth-child(1){
    border-radius: 6px 0 0 6px;
  }
  &:last-child{
    border-radius: 0 6px 6px 0;
  }
}
</style>