<template>
  <div class="t-matrix">
      <div class="t-matrix__gradient">
          <div class="colors"></div>
          <div class="values">
              <div v-for="(item, idx) in values" :key="idx">{{ item }}</div>
          </div>
      </div>
      <div class="t-matrix__table">
          <div class="t-matrix__label-top">{{ label }} – Confusion matrix</div>
          <div class="t-matrix__grid-wrapper">
              <div class="legend-left">
                  <div v-for="val in labels" :key="val">{{ val }}</div>
              </div>
              <div class="t-matrix__grid" :style="{ gridTemplate: `repeat(${labels.length}, 40px) / repeat(${labels.length}, 40px)` }">
                <div class="t-matrix__grid-item"
                v-for="item in 100"
                :key="item"
                :style="{ background: getColor(Math.floor(Math.random() * 41) + 25) }">
                    515 51,5%
                </div>
              </div>
              <div class="legend-bottom">
                  <div v-for="val in labels" :key="val">{{ val }}</div>
              </div>
              <div class="t-matrix__label-left">True label</div>
          </div>
          <div class="t-matrix__label-bottom">Predicted label</div>
      </div>
  </div>
</template>

<script>
export default {
    name: 't-matrix',
    props: ['label'],
    data: () => ({
        values: ['', 100, 200, 300, 400, 500],
        labels: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    }),
    methods: {
        getColor(light) {
            return `hsl(212, 100%, ${light}%)`
        }
    }
}
</script>

<style lang="scss" scoped>
.t-matrix {
    display: flex;
    align-items: center;
    color: #A7BED3;
    width: fit-content;
    &__gradient {
        display: flex;
        gap: 5px;
        .colors {
            background: linear-gradient(180deg, #003B7F 0%, #54A3FF 100%);
            border-radius: 4px;
            width: 24px;
            height: 400px;
            flex-shrink: 0;
        }
        .values {
            display: flex;
            flex-direction: column-reverse;
            justify-content: space-between;
            font-size: 9px;
            line-height: 14px;
        }
    }
    &__table {
        position: relative;
    }
    &__grid {
        display: grid;
        border-radius: 4px;
        overflow: hidden;
        &-wrapper {
            display: flex;
            flex-wrap: wrap;
            max-width: 500px;
            margin: 10px auto;
            justify-content: flex-end;
            .legend-left, .legend-bottom {
                font-size: 9px;
                line-height: 11px;
            }
            .legend-left {
                display: flex;
                flex-direction: column;
                * {
                    flex-basis: 10%;
                    display: flex;
                    justify-content: flex-end;
                    align-items: center;
                    padding: 0 5px;
                }
            }
            .legend-bottom {
                display: flex;
                justify-content: flex-end;
                * {
                    flex: 0 0 40px;
                    word-wrap: break-word;
                    text-align: center;
                    max-width: 40px;
                    padding: 3px 0;
                }
            }
        }
        &-item {
            font-size: 10px;
            line-height: 14px;
            display: flex;
            text-align: center;
            justify-content: center;
            align-items: center;
        }
    }
    &__label-top {
        text-align: center;
        font-size: 14px;
        line-height: 17px;
        font-weight: 600;
    }
    &__label-bottom, &__label-left {
        text-align: center;
        font-weight: 600;
        font-size: 12px;
        line-height: 16px;
    }
    &__label-left {
        position: absolute;
        top: 50%;
        left: 0;
        transform: rotate(-90deg) translateX(50%);
    }
}
</style>