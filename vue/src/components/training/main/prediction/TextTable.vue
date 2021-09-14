<template>
  <div class="text-table" v-if="show">
    <div class="text-table__columns">
      <div class="text-table__column">
        <p>Слой</p>
      </div>
      <div class="text-table__column text-table__column--mainer">
        <p>Истинное значение</p>
        <p>Output_2</p>
      </div>
      <div class="text-table__column text-table__column--mainer">
        <p>Предсказание</p>
        <p>Output_3</p>
      </div>
      <div class="text-table__column">
        <p>Tags color</p>
      </div>
    </div>
    <scrollbar>
      <div class="text-table__body">
        <!-- First Row -->
        <div class="text-table__rows">
          <div class="text-table__row">
            <div class="text-table__content">
              <p>1</p>
            </div>
          </div>
          <div class="text-table__row text-table__row--mainer">
            <div class="text-table__content">
              <p>
                Идейные соображения высшего порядка, а также сложившаяся структура организации играет важную роль в
                формировании направлений прогрессивного развития. Товарищи! постоянное информационно-пропагандистское
                обеспечение нашей деятельности требуют определения и уточнения системы обучения кадров, соответствует
                насущным потребностям. С другой стороны рамки и место обучения кадров играет важную роль в формировании
                существенных финансовых и административных условий. Не следует, однако забывать, что начало повседневной
                работы по формированию позиции в значительной степени обуславливает создание систем массового участия.
                Не следует, однако забывать, что сложившаяся структура организации позволяет выполнять важные задания по
                разработке новых предложений.
              </p>
            </div>
            <div class="text-table__content">
              <p>
                <span class="text-table__content--marked" style="background: #5dbb31">
                  Идейные соображения высшего порядка, а также сложившаяся структура организации играет
                </span>
                важную роль в формировании направлений прогрессивного развития. Товарищи! постоянное
                информационно-пропагандистское обеспечение нашей деятельности требуют определения и уточнения системы
                обучения кадров, соответствует насущным потребностям. С другой стороны рамки и место обучения кадров
                играет важную роль в формировании существенных финансовых и административных условий. Не следует, однако
                забывать, что начало повседневной работы по формированию позиции в значительной степени обуславливает
                создание систем массового участия. Не следует, однако забывать, что сложившаяся структура организации
                позволяет выполнять важные задания по разработке новых предложений.
              </p>
            </div>
          </div>
          <div class="text-table__row text-table__row--mainer">
            <div class="text-table__content">
              <p>
                Идейные соображения высшего порядка, а также сложившаяся структура организации играет важную роль в
                формировании направлений прогрессивного развития. Товарищи! постоянное информационно-пропагандистское
                обеспечение нашей деятельности требуют определения и уточнения системы обучения кадров, соответствует
                насущным потребностям. С другой стороны рамки и место обучения кадров играет важную роль в формировании
                существенных финансовых и административных условий. Не следует, однако забывать, что начало повседневной
                работы по формированию позиции в значительной степени обуславливает создание систем массового участия.
                Не следует, однако забывать, что сложившаяся структура организации позволяет выполнять важные задания по
                разработке новых предложений.
              </p>
            </div>
          </div>
          <div class="text-table__row">
            <div class="text-table__content text-table__content--tags">
              <p>s1</p>
              <div class="text-table__content--tag" style="background: #ffb054"></div>
            </div>
            <div class="text-table__content text-table__content--tags">
              <p>s1</p>
              <div class="text-table__content--tag" style="background: #fe8900"></div>
            </div>
            <div class="text-table__content text-table__content--tags">
              <p>s1</p>
              <div class="text-table__content--tag" style="background: #e21175"></div>
            </div>
          </div>
        </div>

        <!-- Second Row -->
        <div class="text-table__rows">
          <div class="text-table__row">
            <div class="text-table__content">
              <p>2</p>
            </div>
          </div>
          <div class="text-table__row text-table__row--mainer">
            <div class="text-table__content">
              <p>
                Идейные соображения высшего порядка, а также сложившаяся структура организации играет важную роль в
                формировании направлений прогрессивного развития. Товарищи! постоянное информационно-пропагандистское
                обеспечение нашей деятельности требуют определения и уточнения системы обучения кадров, соответствует
                насущным потребностям. С другой стороны рамки и место обучения кадров играет важную роль в формировании
                существенных финансовых и административных условий. Не следует, однако забывать, что начало повседневной
                работы по формированию позиции в значительной степени обуславливает создание систем массового участия.
                Не следует, однако забывать, что сложившаяся структура организации позволяет выполнять важные задания по
                разработке новых предложений.
              </p>
            </div>
            <div class="text-table__content">
              <p>
                <span class="text-table__content--marked" style="background: #5dbb31">
                  Идейные соображения высшего порядка, а также сложившаяся структура организации играет
                </span>
                важную роль в формировании направлений прогрессивного развития. Товарищи! постоянное
                информационно-пропагандистское обеспечение нашей деятельности требуют определения и уточнения системы
                обучения кадров, соответствует насущным потребностям. С другой стороны рамки и место обучения кадров
                играет важную роль в формировании существенных финансовых и административных условий. Не следует, однако
                забывать, что начало повседневной работы по формированию позиции в значительной степени обуславливает
                создание систем массового участия. Не следует, однако забывать, что сложившаяся структура организации
                позволяет выполнять важные задания по разработке новых предложений.
              </p>
            </div>
          </div>
          <div class="text-table__row text-table__row--mainer">
            <div class="text-table__content">
              <p>
                Идейные соображения высшего порядка, а также сложившаяся структура организации играет важную роль в
                формировании направлений прогрессивного развития. Товарищи! постоянное информационно-пропагандистское
                обеспечение нашей деятельности требуют определения и уточнения системы обучения кадров, соответствует
                насущным потребностям. С другой стороны рамки и место обучения кадров играет важную роль в формировании
                существенных финансовых и административных условий. Не следует, однако забывать, что начало повседневной
                работы по формированию позиции в значительной степени обуславливает создание систем массового участия.
                Не следует, однако забывать, что сложившаяся структура организации позволяет выполнять важные задания по
                разработке новых предложений.
              </p>
            </div>
          </div>
          <div class="text-table__row">
            <div class="text-table__content text-table__content--tags">
              <p>s1</p>
              <div class="text-table__content--tag" style="background: #fe8900"></div>
            </div>
            <div class="text-table__content text-table__content--tags">
              <p>s1</p>
              <div class="text-table__content--tag" style="background: #5dbb31"></div>
            </div>
            <div class="text-table__content text-table__content--tags">
              <p>s1</p>
              <div class="text-table__content--tag" style="background: #6011e2"></div>
            </div>
          </div>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
export default {
  name: 'TextTable',
  props: {
    show: Boolean,
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
      height: 44px;
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
        flex: 1 1;
      }
      &:last-child {
        border-right: 0;
        flex: 0 0 138px;
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
        flex: 0 0 138px;
        padding: 10px;
        p {
          font-size: 14px;
          line-height: 24px;
        }
      }
    }
  }
}
</style>
