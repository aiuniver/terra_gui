<template>
  <div class="load__answer">
    <scrollbar :ops="ops">
      <div class="answer__text">
        {{ moduleList }}
      </div>
<!--      <div class="terra-url">https://srv2.demo.neural-university.ru</div>-->
<!--      <div class="answer-url">-->
<!--        Чтобы получилось:-->
<!--        <span>-->
<!--              https://srv2.demo.neural-university.ru/media/check_od/output_files/34234452.jpg-->
<!--            </span>-->
<!--      </div>-->
<!--      <div class="module-list">-->
<!--        <div class="module-list__title">СПИСОК МОДУЛЕЙ:</div>-->
<!--        <div class="module-list__item" v-for="item in moduleList" :key="item">-->
<!--          <div class="item__title">{{ item.title }}</div>-->
<!--          <div class="item__url">{{ item.url }}</div>-->
<!--          <div class="item__input">-->
<!--            Вход:-->
<!--            <span>{{ item.input }}</span>-->
<!--          </div>-->
<!--          <div class="item__output">-->
<!--            Выход:-->
<!--            <span>{{ item.output }}</span>-->
<!--          </div>-->
<!--        </div>-->
<!--      </div>-->
    </scrollbar>
  </div>
</template>

<script>
export default {
  name: "ModuleList",
  props: {
    moduleList: {
      type: String,
      default: () => ""
    }
  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  })
}
</script>

<style scoped lang="scss">
.load__answer{
  height: 464px;
  padding: 16px;
  margin-top: 20px;
  background: #242F3D;
  border: 1px solid #6C7883;
  box-sizing: border-box;
  border-radius: 4px;
}
.answer{
  &__text{
    width: 325px;
    height: 430px;
  }
}
.terra-url{
  color: #65B9F4;
  font-size: 14px;
  line-height: 21px;
  font-style: italic;
  padding-bottom: 20px;
}
.answer-url{
  font-size: 14px;
  line-height: 21px;
  span{
    padding-top: 20px;
    display: block;
    font-size: 14px;
    line-height: 21px;
    font-style: italic;
    color: #65B9F4;
  }
}
.module-list{
  padding-top: 40px;
}
.module-list__title{
  font-size: 16px;
  line-height: 21px;
  font-weight: 700;
  padding-bottom: 10px;
}
.module-list__item{
  padding: 10px 0;
}
.item__title{
  font-size: 14px;
  line-height: 21px;
  font-weight: 700;
}
.item__url{
  color: #65B9F4;
  font-size: 14px;
  line-height: 21px;
  font-style: italic;
}
.item__input, .item__output{
  span{
    display: block;
  }
}
</style>