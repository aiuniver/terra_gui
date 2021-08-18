<template>
  <div class="content">
    <table class="table">
      <tr class="table__title-row">
        <td>
          <button class="table__reload-all">
            <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
            <span>Перезагрузить все</span>
          </button>
        </td>
        <td>Предсказанные данные</td>
        <td>Комнат</td>
        <td>Метро / ЖД станции</td>
        <td>От станции</td>
        <td>Дом</td>
        <td>Балкон</td>
        <td>Санузел</td>
        <td>Площадь</td>
        <td>Цена, руб.</td>
        <td>ГРМ</td>
        <td>Бонус агенству</td>
      </tr>
      <tr>
        <td class="table__td-reload">
          <button class="td-reload__btn-reload">
            <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
          </button>
        </td>
        <td class="table__result-data">Мастер и Маргарита, Михаил Афанасьевич Булгаков </td>
        <td>1</td>
        <td>Шелепиха м.</td>
        <td>12п</td>
        <td>35/37 М</td>
        <td>NaN</td>
        <td>2</td>
        <td>64.1/23/20</td>
        <td>19500000.0</td>
        <td>NaN</td>
        <td>NaN</td>
      </tr>
       <tr>
        <td class="table__td-reload">
          <button class="td-reload__btn-reload">
            <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
          </button>
        </td>
        <td class="table__result-data">Мастер и Маргарита, Михаил Афанасьевич Булгаков </td>
        <td>1</td>
        <td>Шелепиха м.</td>
        <td>12п</td>
        <td>35/37 М</td>
        <td>NaN</td>
        <td>2</td>
        <td>64.1/23/20</td>
        <td>19500000.0</td>
        <td>NaN</td>
        <td>NaN</td>
      </tr>
    </table>
  </div>
</template>

<script>
export default {
  name: "Table"
}
</script>

<style scoped lang="scss">
.table{
  width: 100%;
  background: #242F3D;
  border: 1px solid #6C7883;
  box-sizing: border-box;
  border-radius: 4px;
  tr:nth-child(even){
    background: #17212B;
  }
  td{
    padding: 5px;
  }
}
.table__title-row{
  color: #FFFFFF;
  font-weight: bold;
  font-size: 14px;
  line-height: 24px;
  td{
    text-align: left;
    &:nth-child(1){
      padding-left: 8px;
    }
  }
}
.table__result-data{
  width: 200px;
  color: #65B9F4;
}
.table__reload-all{
  display: flex;
  width: 174px;
  padding: 8px 10px 10px 10px;
  justify-content: center;
  align-items: center;
  i{
    width: 16px;
  }
  span{
    font-weight: normal;
    font-size: 14px;
    line-height: 24px;
    padding-left: 8px;
  }
}
.table__td-reload{
  text-align: center;
}
.td-reload__btn-reload{
  display: flex;
  justify-content: center;
  align-items: center;
  width: 32px;
  height: 32px;
  padding-bottom: 2px;
  i{
    width: 16px;
  }
}
</style>