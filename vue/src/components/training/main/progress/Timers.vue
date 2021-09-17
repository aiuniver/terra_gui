<template>
  <div class="t-time">
    <div class="t-time__train">
      <div class="t-time__timer-wrapper">
        <div class="t-time__timer">
          <span>Расчетное время обучения</span>
          <div>0h : 55m : 07s</div>
        </div>
        <div class="t-time__timer">
          <span>Прошло времени</span>
          <div>0h : 35m : 07s</div>
        </div>
        <div class="t-time__timer">
          <span>Время до окончания обучения</span>
          <div>0h : 07m : 35s</div>
        </div>
        <div class="t-time__timer">
          <span>Эпоха</span>
          <div>18</div>
          /
          <div>25</div>
        </div>
      </div>
      <div class="t-time__progress-bar">
        <div class="t-time__progress-bar--fill" style="width: 553px"></div>
      </div>
    </div>
    <div class="t-time__age">
      <div class="t-time__timer-wrapper">
        <div class="t-time__timer">
          <span>Среднее время эпохи</span>
          <div>0h : 05m : 57s</div>
        </div>
        <div class="t-time__timer">
          <span>Прошло времени на эпоху</span>
          <div>0h : 05m : 37s</div>
        </div>
        <div class="t-time__timer">
          <span>Время до окончания текущей эпохи</span>
          <div>0h : 00m : 15s</div>
        </div>
        <div class="t-time__timer">
          <span>Батч</span>
          <div>245</div>
          /
          <div>256</div>
        </div>
      </div>
      <div class="t-time__progress-bar">
        <div class="t-time__progress-bar--fill" style="width: 743px"></div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-train-time',
};
</script>

<style lang="scss" scoped>
.t-time {
  width: 100%;
  &__timer-wrapper {
    display: flex;
    gap: 20px;
  }
  &__progress-bar {
    background: #2b5278;
    border-radius: 5px;
    height: 10px;
    margin-top: 10px;
    &--fill {
      background: #65b9f4;
      height: 100%;
      border-radius: 5px;
    }
  }
  &__age {
    margin-top: 20px;
  }
  &__timer {
    &:first-child div {
      border: 1px solid #65b9f4;
    }
    &:nth-child(2) div {
      border: 1px solid #fff;
    }
    &:last-child {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      text-align: center;
      justify-content: space-between;
      flex-shrink: 0;
      flex-grow: 0;
      width: min-content;
      span {
        width: 92px;
        text-align: center;
      }
      div {
        width: 40px;
        text-align: center;
        padding: 8px 0;
      }
    }
    div {
      font-size: 14px;
      color: #ffffff;
      padding: 8px 10px;
      background: #242f3d;
      border: 1px solid #6c7883;
      border-radius: 10px;
      width: 212px;
      text-align: center;
      margin-top: 10px;
    }
  }
}
</style>