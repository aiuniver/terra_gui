<template>
  <div class="table" v-if="show">
    <div class="table__columns">
      <div class="table__column">
        <div class="table__item title size-1">Слой</div>
        <div class="table__item">1</div>
        <div class="table__item">2</div>
        <div class="table__item">3</div>
        <div class="table__item">4</div>
      </div>
      <div class="table__column">
        <div class="table__item">
          <div class="table__column">
            <div class="table__item title size-2">
              Исходные данные
            </div>
            <div class="table__row">
              <div class="table__column">
                <div class="table__item title size-2">Вход 1</div>
                <div class="table__item">FROG</div>
                <div class="table__item">TRUCK</div>
                <div class="table__item">SHIP</div>
                <div class="table__item">FROG</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="table__column">
        <div class="table__item">
          <div class="table__column">
            <div class="table__item title size-2">
              Истинное значение
            </div>
            <div class="table__row">
              <div class="table__column">
                <div class="table__item title size-2">Выход 2</div>
                <div class="table__item">FROG</div>
                <div class="table__item">TRUCK</div>
                <div class="table__item">SHIP</div>
                <div class="table__item">FROG</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="table__column">
        <div class="table__item">
          <div class="table__column">
            <div class="table__column title size-2">
              Предсказание
            </div>
            <div class="table__row">
              <div class="table__column">
                <div class="table__item title size-2">Выход 3</div>
                <div class="table__item">FROG</div>
                <div class="table__item">TRUCK</div>
                <div class="table__item">SHIP</div>
                <div class="table__item">FROG</div>
              </div>
            </div>
          </div>
        </div>

      </div>
      <div class="table__column">
        <div class="table__item">
          <div class="table__column">
            <div class="table__item title size-2">
              Статистика примеров
            </div>
            <div class="table__row">
              <div class="table__column">
                <div class="table__item title size-2">airplane</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
              </div>
              <div class="table__column">
                <div class="table__item title size-2">frog</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
              </div>
              <div class="table__column">
                <div class="table__item title size-2">truck</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
              </div>
              <div class="table__column">
                <div class="table__item title size-2">ship</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
              </div>
              <div class="table__column">
                <div class="table__item title size-2">horse</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
                <div class="table__item">0.9%</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TextTableTest',
  props: {
    show: Boolean,
    predict: {
      type: Object,
      default: ()=>({})
    }
  },
  data: () => ({
  }),
};
</script>

<style lang="scss" scoped>
$bgGray: #242f3d;
$border: #0e1621;

.table{
  width: 100%;
  border: 1px solid $bgGray;
  border-radius: 3px;
  &__columns{
    display: flex;
    flex-direction: row;
  }
  &__row{
    display: flex;
  }
  &__column{
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  &__item{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
  }
  .title{
    background: $bgGray;
    min-width: 70px;
  }
}

//.text-table {
//  width: 100%;
//  border: 1px solid $bgGray;
//  border-radius: 3px;
//
//  &__body {
//    max-height: 500px;
//  }
//
//  &__content {
//    &--tag {
//      width: 77px;
//      height: 24px;
//      border-radius: 4px;
//    }
//    &--tags {
//      &:first-child {
//        margin-top: 0px;
//      }
//      margin-top: 10px;
//      display: flex;
//      align-items: center;
//      justify-content: space-between;
//      margin-right: 5px;
//      p {
//        margin-right: 10px;
//      }
//    }
//  }
//
//  &__columns {
//    display: flex;
//    width: 100%;
//    .text-table__column {
//      text-align: center;
//      //height: 44px;
//      background: $bgGray;
//      border-right: 1px solid $border;
//      p {
//        font-family: 'Open Sans';
//        font-style: normal;
//        font-weight: 600;
//        font-size: 12px;
//        line-height: 16px;
//        color: #a7bed3;
//      }
//      &:first-child {
//        flex: 0 0 51px;
//        display: flex;
//        justify-content: center;
//        align-items: center;
//      }
//      &--mainer {
//        display: flex;
//        flex-direction: column;
//        p {
//          height: 100%;
//          line-height: 22px;
//          border-bottom: 1px solid $border;
//          &:last-child {
//            border-bottom: 0;
//          }
//        }
//        //flex: 1 1;
//      }
//      &:last-child {
//        border-right: 0;
//        //flex: 0 0 138px;
//        display: flex;
//        justify-content: center;
//        align-items: center;
//      }
//    }
//  }
//
//  &__rows {
//    display: flex;
//    width: 100%;
//    border-top: 1px solid $border;
//    &:first-child {
//      border-top: 0;
//    }
//    .text-table__row {
//      border-right: 1px solid $border;
//
//      p {
//        font-family: 'Open Sans';
//        font-style: normal;
//        font-weight: normal;
//        font-size: 12px;
//        line-height: 18px;
//        min-width: 18px;
//        /* or 125% */
//      }
//      &:first-child {
//        flex: 0 0 51px;
//        display: flex;
//        justify-content: center;
//        align-items: center;
//      }
//      &--mainer {
//        padding: 10px;
//        .text-table__content {
//          margin-top: 10px;
//          &--marked {
//            mix-blend-mode: lighten;
//            border-radius: 4px;
//          }
//          &:first-child {
//            margin-top: 0;
//          }
//        }
//      }
//      &:last-child {
//        border-right: 0;
//        //flex: 0 0 138px;
//        padding: 10px;
//        p {
//          font-size: 14px;
//          line-height: 24px;
//        }
//      }
//    }
//  }
  .grow-normal{
    flex-grow: 1;
  }
  .grow-large{
    flex-grow: 2;
  }
  .grow-extra{
    flex-grow: 3;
  }
  .align-center{
    align-items: center;
  }
  .justify-center{
    justify-content: center;
  }
  .size-1{
    height: 46px;
  }
  .size-2{
    height: 23px;
  }
</style>
