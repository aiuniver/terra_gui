<template>
  <div class="text-table" v-if="show">
    <div class="text-table__columns">
      <div class="text-table__column">
        <p>Слой</p>
      </div>
      <div class="text-table__column text-table__column--mainer grow-normal" v-if="predict.initial_value">
        <p>Исходные данные</p>
        <p v-for="(value, input_name) in predict.initial_value" :key="input_name">{{ input_name }}</p>
      </div>
      <div class="text-table__column text-table__column--mainer grow-normal" v-if="predict.true_value">
        <p>Истинное значение</p>
        <p v-for="(value, output_name) in predict.true_value" :key="output_name">{{ output_name }}</p>
      </div>
      <div class="text-table__column text-table__column--mainer grow-normal" v-if="predict.predict_value">
        <p>Предсказание</p>
        <p v-for="(value, output_name) in predict.predict_value" :key="output_name">{{ output_name }}</p>
      </div>
      <div class="text-table__column text-table__column--mainer grow-large" v-if="predict.statistic_values">
        <p>Статистика примеров</p>
        <div class="text-table__rows align-center">
          <div class="text-table__row grow-normal" v-for="(value, class_name) in predict.statistic_values" :key="class_name">
            <p>{{ class_name }}</p>
          </div>
        </div>
      </div>

    </div>
    <scrollbar>
      <div class="text-table__body">
        <div class="text-table__columns">
          <div class="text-table__column text-table__column--mainer">
            <p v-for="index in Object.keys(predict.initial_value[Object.keys(predict.initial_value)[0]]).length" :key="index">
              {{ index }}
            </p>
          </div>
          <div class="text-table__column text-table__column--mainer" v-for="(input, input_index) in predict.initial_value" :key="input_index">
            <img v-for="(data, index) in input" :src="require('@/../public/imgs/'+data.data)" :alt="'img'" :key="index" width="270">
          </div>
          <div class="text-table__column text-table__column--mainer grow-normal" v-for="(output, output_index) in predict.true_value" :key="output_index">
            <p v-for="(data, index) in output" :key="index">
              {{ data.data }}
            </p>
          </div>
          <div class="text-table__column text-table__column--mainer grow-normal" v-for="(output, output_index) in predict.predict_value" :key="output_index">
            <p v-for="(data, index) in output" :key="index">
              {{ data.data }}
            </p>
          </div>

        </div>
      </div>
    </scrollbar>
<!--    <scrollbar>-->
<!--      <div class="text-table__body">-->
<!--        &lt;!&ndash; First Row &ndash;&gt;-->
<!--        <div class="text-table__rows">-->
<!--          <div class="text-table__row">-->
<!--            <div class="text-table__content">-->
<!--              <p>1</p>-->
<!--            </div>-->
<!--          </div>-->
<!--          <div class="text-table__row text-table__row&#45;&#45;mainer">-->
<!--            <div class="text-table__content">-->
<!--              <p>-->
<!--                Идейные соображения высшего порядка, а также сложившаяся структура организации играет важную роль в-->
<!--                формировании направлений прогрессивного развития. Товарищи! постоянное информационно-пропагандистское-->
<!--                обеспечение нашей деятельности требуют определения и уточнения системы обучения кадров, соответствует-->
<!--                насущным потребностям. С другой стороны рамки и место обучения кадров играет важную роль в формировании-->
<!--                существенных финансовых и административных условий. Не следует, однако забывать, что начало повседневной-->
<!--                работы по формированию позиции в значительной степени обуславливает создание систем массового участия.-->
<!--                Не следует, однако забывать, что сложившаяся структура организации позволяет выполнять важные задания по-->
<!--                разработке новых предложений.-->
<!--              </p>-->
<!--            </div>-->
<!--            <div class="text-table__content">-->
<!--              <p>-->
<!--                <span class="text-table__content&#45;&#45;marked" style="background: #5dbb31">-->
<!--                  Идейные соображения высшего порядка, а также сложившаяся структура организации играет-->
<!--                </span>-->
<!--                важную роль в формировании направлений прогрессивного развития. Товарищи! постоянное-->
<!--                информационно-пропагандистское обеспечение нашей деятельности требуют определения и уточнения системы-->
<!--                обучения кадров, соответствует насущным потребностям. С другой стороны рамки и место обучения кадров-->
<!--                играет важную роль в формировании существенных финансовых и административных условий. Не следует, однако-->
<!--                забывать, что начало повседневной работы по формированию позиции в значительной степени обуславливает-->
<!--                создание систем массового участия. Не следует, однако забывать, что сложившаяся структура организации-->
<!--                позволяет выполнять важные задания по разработке новых предложений.-->
<!--              </p>-->
<!--            </div>-->
<!--          </div>-->
<!--          <div class="text-table__row text-table__row&#45;&#45;mainer">-->
<!--            <div class="text-table__content">-->
<!--              <p>-->
<!--                Идейные соображения высшего порядка, а также сложившаяся структура организации играет важную роль в-->
<!--                формировании направлений прогрессивного развития. Товарищи! постоянное информационно-пропагандистское-->
<!--                обеспечение нашей деятельности требуют определения и уточнения системы обучения кадров, соответствует-->
<!--                насущным потребностям. С другой стороны рамки и место обучения кадров играет важную роль в формировании-->
<!--                существенных финансовых и административных условий. Не следует, однако забывать, что начало повседневной-->
<!--                работы по формированию позиции в значительной степени обуславливает создание систем массового участия.-->
<!--                Не следует, однако забывать, что сложившаяся структура организации позволяет выполнять важные задания по-->
<!--                разработке новых предложений.-->
<!--              </p>-->
<!--            </div>-->
<!--          </div>-->
<!--&lt;!&ndash;          <div class="text-table__row">&ndash;&gt;-->
<!--&lt;!&ndash;            <div class="text-table__content text-table__content&#45;&#45;tags">&ndash;&gt;-->
<!--&lt;!&ndash;              <p>s1</p>&ndash;&gt;-->
<!--&lt;!&ndash;              <div class="text-table__content&#45;&#45;tag" style="background: #ffb054"></div>&ndash;&gt;-->
<!--&lt;!&ndash;            </div>&ndash;&gt;-->
<!--&lt;!&ndash;            <div class="text-table__content text-table__content&#45;&#45;tags">&ndash;&gt;-->
<!--&lt;!&ndash;              <p>s1</p>&ndash;&gt;-->
<!--&lt;!&ndash;              <div class="text-table__content&#45;&#45;tag" style="background: #fe8900"></div>&ndash;&gt;-->
<!--&lt;!&ndash;            </div>&ndash;&gt;-->
<!--&lt;!&ndash;            <div class="text-table__content text-table__content&#45;&#45;tags">&ndash;&gt;-->
<!--&lt;!&ndash;              <p>s1</p>&ndash;&gt;-->
<!--&lt;!&ndash;              <div class="text-table__content&#45;&#45;tag" style="background: #e21175"></div>&ndash;&gt;-->
<!--&lt;!&ndash;            </div>&ndash;&gt;-->
<!--&lt;!&ndash;          </div>&ndash;&gt;-->
<!--        </div>-->

<!--        &lt;!&ndash; Second Row &ndash;&gt;-->
<!--        <div class="text-table__rows">-->
<!--          <div class="text-table__row">-->
<!--            <div class="text-table__content">-->
<!--              <p>2</p>-->
<!--            </div>-->
<!--          </div>-->
<!--          <div class="text-table__row text-table__row&#45;&#45;mainer">-->
<!--            <div class="text-table__content">-->
<!--              <p>-->
<!--                Идейные соображения высшего порядка, а также сложившаяся структура организации играет важную роль в-->
<!--                формировании направлений прогрессивного развития. Товарищи! постоянное информационно-пропагандистское-->
<!--                обеспечение нашей деятельности требуют определения и уточнения системы обучения кадров, соответствует-->
<!--                насущным потребностям. С другой стороны рамки и место обучения кадров играет важную роль в формировании-->
<!--                существенных финансовых и административных условий. Не следует, однако забывать, что начало повседневной-->
<!--                работы по формированию позиции в значительной степени обуславливает создание систем массового участия.-->
<!--                Не следует, однако забывать, что сложившаяся структура организации позволяет выполнять важные задания по-->
<!--                разработке новых предложений.-->
<!--              </p>-->
<!--            </div>-->
<!--            <div class="text-table__content">-->
<!--              <p>-->
<!--                <span class="text-table__content&#45;&#45;marked" style="background: #5dbb31">-->
<!--                  Идейные соображения высшего порядка, а также сложившаяся структура организации играет-->
<!--                </span>-->
<!--                важную роль в формировании направлений прогрессивного развития. Товарищи! постоянное-->
<!--                информационно-пропагандистское обеспечение нашей деятельности требуют определения и уточнения системы-->
<!--                обучения кадров, соответствует насущным потребностям. С другой стороны рамки и место обучения кадров-->
<!--                играет важную роль в формировании существенных финансовых и административных условий. Не следует, однако-->
<!--                забывать, что начало повседневной работы по формированию позиции в значительной степени обуславливает-->
<!--                создание систем массового участия. Не следует, однако забывать, что сложившаяся структура организации-->
<!--                позволяет выполнять важные задания по разработке новых предложений.-->
<!--              </p>-->
<!--            </div>-->
<!--          </div>-->
<!--          <div class="text-table__row text-table__row&#45;&#45;mainer">-->
<!--            <div class="text-table__content">-->
<!--              <p>-->
<!--                Идейные соображения высшего порядка, а также сложившаяся структура организации играет важную роль в-->
<!--                формировании направлений прогрессивного развития. Товарищи! постоянное информационно-пропагандистское-->
<!--                обеспечение нашей деятельности требуют определения и уточнения системы обучения кадров, соответствует-->
<!--                насущным потребностям. С другой стороны рамки и место обучения кадров играет важную роль в формировании-->
<!--                существенных финансовых и административных условий. Не следует, однако забывать, что начало повседневной-->
<!--                работы по формированию позиции в значительной степени обуславливает создание систем массового участия.-->
<!--                Не следует, однако забывать, что сложившаяся структура организации позволяет выполнять важные задания по-->
<!--                разработке новых предложений.-->
<!--              </p>-->
<!--            </div>-->
<!--          </div>-->
<!--&lt;!&ndash;          <div class="text-table__row">&ndash;&gt;-->
<!--&lt;!&ndash;            <div class="text-table__content text-table__content&#45;&#45;tags">&ndash;&gt;-->
<!--&lt;!&ndash;              <p>s1</p>&ndash;&gt;-->
<!--&lt;!&ndash;              <div class="text-table__content&#45;&#45;tag" style="background: #fe8900"></div>&ndash;&gt;-->
<!--&lt;!&ndash;            </div>&ndash;&gt;-->
<!--&lt;!&ndash;            <div class="text-table__content text-table__content&#45;&#45;tags">&ndash;&gt;-->
<!--&lt;!&ndash;              <p>s1</p>&ndash;&gt;-->
<!--&lt;!&ndash;              <div class="text-table__content&#45;&#45;tag" style="background: #5dbb31"></div>&ndash;&gt;-->
<!--&lt;!&ndash;            </div>&ndash;&gt;-->
<!--&lt;!&ndash;            <div class="text-table__content text-table__content&#45;&#45;tags">&ndash;&gt;-->
<!--&lt;!&ndash;              <p>s1</p>&ndash;&gt;-->
<!--&lt;!&ndash;              <div class="text-table__content&#45;&#45;tag" style="background: #6011e2"></div>&ndash;&gt;-->
<!--&lt;!&ndash;            </div>&ndash;&gt;-->
<!--&lt;!&ndash;          </div>&ndash;&gt;-->
<!--        </div>-->
<!--      </div>-->
<!--    </scrollbar>-->
  </div>
</template>

<script>
export default {
  name: 'TextTable',
  props: {
    show: Boolean,
    predict: {
      type: Object,
      default: ()=>({})
    }
  },
  data: () => ({
    // ops: {
    //   scrollPanel: {
    //     scrollingX: true,
    //     scrollingY: false,
    //   },
    //   rail: {
    //     gutterOfEnds: '6px',
    //   },
    // },
  }),
};
</script>

<style lang="scss" scoped>
$bgGray: #242f3d;
$border: #0e1621;

.text-table {
  width: 100%;
  border: 1px solid $bgGray;
  border-radius: 3px;

  &__body {
    max-height: 500px;
  }

  &__content {
    &--tag {
      width: 77px;
      height: 24px;
      border-radius: 4px;
    }
    &--tags {
      &:first-child {
        margin-top: 0px;
      }
      margin-top: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-right: 5px;
      p {
        margin-right: 10px;
      }
    }
  }

  &__columns {
    display: flex;
    width: 100%;
    .text-table__column {
      text-align: center;
      //height: 44px;
      background: $bgGray;
      border-right: 1px solid $border;
      p {
        font-family: 'Open Sans';
        font-style: normal;
        font-weight: 600;
        font-size: 12px;
        line-height: 16px;
        color: #a7bed3;
      }
      &:first-child {
        flex: 0 0 51px;
        display: flex;
        justify-content: center;
        align-items: center;
      }
      &--mainer {
        display: flex;
        flex-direction: column;
        p {
          height: 100%;
          line-height: 22px;
          border-bottom: 1px solid $border;
          &:last-child {
            border-bottom: 0;
          }
        }
        //flex: 1 1;
      }
      &:last-child {
        border-right: 0;
        //flex: 0 0 138px;
        display: flex;
        justify-content: center;
        align-items: center;
      }
    }
  }

  &__rows {
    display: flex;
    width: 100%;
    border-top: 1px solid $border;
    &:first-child {
      border-top: 0;
    }
    .text-table__row {
      border-right: 1px solid $border;

      p {
        font-family: 'Open Sans';
        font-style: normal;
        font-weight: normal;
        font-size: 12px;
        line-height: 18px;
        min-width: 18px;
        /* or 125% */
      }
      &:first-child {
        flex: 0 0 51px;
        display: flex;
        justify-content: center;
        align-items: center;
      }
      &--mainer {
        padding: 10px;
        .text-table__content {
          margin-top: 10px;
          &--marked {
            mix-blend-mode: lighten;
            border-radius: 4px;
          }
          &:first-child {
            margin-top: 0;
          }
        }
      }
      &:last-child {
        border-right: 0;
        //flex: 0 0 138px;
        padding: 10px;
        p {
          font-size: 14px;
          line-height: 24px;
        }
      }
    }
  }
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
}
</style>
