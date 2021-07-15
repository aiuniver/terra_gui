<template>
  <div class="texts">
    <div class="category-title">Текст</div>
    <div class="content">
      <div class="inner">
        <div class="epochs">
          <table>
            <thead>
              <tr class="outputs_heads">
                <th rowspan="2">Эпоха</th>
                <th rowspan="2">Время<br />(сек.)</th>
                <th colspan="4">output_1</th>
              </tr>
              <tr class="callbacks_heads">
                <th>accuracy</th>
                <th>val_accuracy</th>
                <th>loss</th>
                <th>val_loss</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="epoch_num">1</td>
                <td>37</td>
                <td class="value"><span>0</span><i>.</i>140</td>
                <td class="value"><span>0</span><i>.</i>244</td>
                <td class="value"><span>3</span><i>.</i>706</td>
                <td class="value"><span>3</span><i>.</i>155</td>
              </tr>
              <tr>
                <td class="epoch_num">2</td>
                <td>6</td>
                <td class="value"><span>0</span><i>.</i>242</td>
                <td class="value"><span>0</span><i>.</i>299</td>
                <td class="value"><span>3</span><i>.</i>093</td>
                <td class="value"><span>2</span><i>.</i>873</td>
              </tr>
              <tr>
                <td class="epoch_num">3</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>285</td>
                <td class="value"><span>0</span><i>.</i>319</td>
                <td class="value"><span>2</span><i>.</i>870</td>
                <td class="value"><span>2</span><i>.</i>756</td>
              </tr>
              <tr>
                <td class="epoch_num">4</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>316</td>
                <td class="value"><span>0</span><i>.</i>325</td>
                <td class="value"><span>2</span><i>.</i>713</td>
                <td class="value"><span>2</span><i>.</i>708</td>
              </tr>
              <tr>
                <td class="epoch_num">5</td>
                <td>6</td>
                <td class="value"><span>0</span><i>.</i>337</td>
                <td class="value"><span>0</span><i>.</i>338</td>
                <td class="value"><span>2</span><i>.</i>601</td>
                <td class="value"><span>2</span><i>.</i>656</td>
              </tr>
              <tr>
                <td class="epoch_num">6</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>357</td>
                <td class="value"><span>0</span><i>.</i>343</td>
                <td class="value"><span>2</span><i>.</i>488</td>
                <td class="value"><span>2</span><i>.</i>650</td>
              </tr>
              <tr>
                <td class="epoch_num">7</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>374</td>
                <td class="value"><span>0</span><i>.</i>346</td>
                <td class="value"><span>2</span><i>.</i>396</td>
                <td class="value"><span>2</span><i>.</i>628</td>
              </tr>
              <tr>
                <td class="epoch_num">8</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>389</td>
                <td class="value"><span>0</span><i>.</i>350</td>
                <td class="value"><span>2</span><i>.</i>324</td>
                <td class="value"><span>2</span><i>.</i>626</td>
              </tr>
              <tr>
                <td class="epoch_num">9</td>
                <td>6</td>
                <td class="value"><span>0</span><i>.</i>411</td>
                <td class="value"><span>0</span><i>.</i>352</td>
                <td class="value"><span>2</span><i>.</i>237</td>
                <td class="value"><span>2</span><i>.</i>613</td>
              </tr>
              <tr>
                <td class="epoch_num">10</td>
                <td>6</td>
                <td class="value"><span>0</span><i>.</i>420</td>
                <td class="value"><span>0</span><i>.</i>338</td>
                <td class="value"><span>2</span><i>.</i>172</td>
                <td class="value"><span>2</span><i>.</i>648</td>
              </tr>
              <tr>
                <td class="epoch_num">11</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>434</td>
                <td class="value"><span>0</span><i>.</i>355</td>
                <td class="value"><span>2</span><i>.</i>109</td>
                <td class="value"><span>2</span><i>.</i>633</td>
              </tr>
              <tr>
                <td class="epoch_num">12</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>450</td>
                <td class="value"><span>0</span><i>.</i>350</td>
                <td class="value"><span>2</span><i>.</i>041</td>
                <td class="value"><span>2</span><i>.</i>646</td>
              </tr>
              <tr>
                <td class="epoch_num">13</td>
                <td>6</td>
                <td class="value"><span>0</span><i>.</i>462</td>
                <td class="value"><span>0</span><i>.</i>353</td>
                <td class="value"><span>1</span><i>.</i>987</td>
                <td class="value"><span>2</span><i>.</i>661</td>
              </tr>
              <tr>
                <td class="epoch_num">14</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>474</td>
                <td class="value"><span>0</span><i>.</i>358</td>
                <td class="value"><span>1</span><i>.</i>932</td>
                <td class="value"><span>2</span><i>.</i>655</td>
              </tr>
              <tr>
                <td class="epoch_num">15</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>482</td>
                <td class="value"><span>0</span><i>.</i>349</td>
                <td class="value"><span>1</span><i>.</i>880</td>
                <td class="value"><span>2</span><i>.</i>705</td>
              </tr>
              <tr>
                <td class="epoch_num">16</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>494</td>
                <td class="value"><span>0</span><i>.</i>348</td>
                <td class="value"><span>1</span><i>.</i>831</td>
                <td class="value"><span>2</span><i>.</i>697</td>
              </tr>
              <tr>
                <td class="epoch_num">17</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>504</td>
                <td class="value"><span>0</span><i>.</i>346</td>
                <td class="value"><span>1</span><i>.</i>788</td>
                <td class="value"><span>2</span><i>.</i>750</td>
              </tr>
              <tr>
                <td class="epoch_num">18</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>512</td>
                <td class="value"><span>0</span><i>.</i>355</td>
                <td class="value"><span>1</span><i>.</i>747</td>
                <td class="value"><span>2</span><i>.</i>763</td>
              </tr>
              <tr>
                <td class="epoch_num">19</td>
                <td>5</td>
                <td class="value"><span>0</span><i>.</i>522</td>
                <td class="value"><span>0</span><i>.</i>349</td>
                <td class="value"><span>1</span><i>.</i>714</td>
                <td class="value"><span>2</span><i>.</i>778</td>
              </tr>
              <tr>
                <td class="epoch_num">20</td>
                <td>6</td>
                <td class="value"><span>0</span><i>.</i>531</td>
                <td class="value"><span>0</span><i>.</i>343</td>
                <td class="value"><span>1</span><i>.</i>670</td>
                <td class="value"><span>2</span><i>.</i>751</td>
              </tr>
            </tbody>
            <tfoot>
              <tr>
                <th colspan="6">Затрачено времени на обучение: 2 мин 53 сек</th>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Texts'
}
</script>