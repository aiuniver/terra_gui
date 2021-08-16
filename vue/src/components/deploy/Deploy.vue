<template>
  <div class="board">
    <div class="wrapper">
      <div class="content">
        <button class="board__reload-all">Перезагрузить все</button>
        <div class="board__data-field">
          <div class="board__title">Исходные данные / Предсказанные данные</div>
          <div class="board__data">
            <IndexCard v-for="(card, i) in Cards" :key="'card-'+i" v-bind="card"/>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import IndexCard from "./IndexCard";

export default {
  components: {
    IndexCard
  },
  data: () => ({
    loaded: false,
    Cards: [
      {
        original: {
          type: "text",
          data: "В белом плаще с кровавым подбоем, шаркающей кавалерийской походкой, ранним утром четырнадцатого числа весеннего месяца нисана в крытую колоннаду между двумя крыльями дворца ирода великого вышел прокуратор Иудеи Понтий Пилат.\n" +
            "\n" +
            "Более всего на свете прокуратор ненавидел запах розового масла, и все теперь предвещало нехороший день, так как запах этот начал преследовать прокуратора с рассвета. Прокуратору казалось, что розовый запах источают кипарисы и пальмы в саду, что к запаху кожи и конвоя примешивается проклятая розовая струя. От флигелей в тылу дворца, где расположилась пришедшая с прокуратором в Ершалаим первая когорта двенадцатого молниеносного легиона, заносило дымком в колоннаду через верхнюю площадку сада, и к горьковатому дыму, свидетельствовавшему о том, что кашевары в кентуриях начали готовить обед, примешивался все тот же жирный розовый дух. О боги, боги, за что вы наказываете меня?" +
            "В белом плаще с кровавым подбоем, шаркающей кавалерийской походкой, ранним утром четырнадцатого числа весеннего месяца нисана в крытую колоннаду между двумя крыльями дворца ирода великого вышел прокуратор Иудеи Пон"
            + "asdfmskdjfhsdkjfhlsdfjksdhfjksdhjksdfhfjksdofgsdjukfjio'sh"
        },
        result: {
          type: "text",
          data: "Мастер и Маргарита, Михаил Афанасьевич Булгаков"
        }
      },
    ]
  }),
  computed: {
    ...mapGetters({
      dataLoaded: "deploy/getDataLoaded",
      height: "settings/autoHeight",
    }),
  },
  methods: {
    click(dataset){
      console.log(dataset)
    }
  },
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 1;
  width: 100%;
}
.wrapper{
  padding: 50px;
  height: 100%;
}
.board__load-data{
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  button{
    width: 156px;
    font-size: 14px;
    line-height: 24px;
  }
}
.board__reload-all{
  width: 174px;
}
.board__data-field{
  padding-top: 30px;
}
.board__title{
    font-size: 12px;
    line-height: 24px;
    color: #A7BED3;
  }
.board__data{
  display: flex;
  flex-wrap: wrap;
}

</style>